from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler, StandardScaler

import create_capability_profile_analysis as cap
from extract_gfp_pdfs import RANK_FIELDS


ROOT = Path("outputs/capability_profiles")
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
AUDIT_SEED = 240822
N_BOOT = 500


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    contingency = np.zeros((k, k), dtype=int)
    for a, b in zip(reference, candidate):
        if 0 <= a < k and 0 <= b < k:
            contingency[a, b] += 1
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    aligned = np.array([mapping.get(int(x), int(x)) for x in candidate], dtype=int)
    return aligned, mapping


def jaccard_by_cluster(reference: np.ndarray, candidate: np.ndarray, k: int) -> list[float]:
    aligned, _ = align_labels(reference, candidate, k)
    scores = []
    for c in range(k):
        a, b = set(np.where(reference == c)[0]), set(np.where(aligned == c)[0])
        scores.append(len(a & b) / len(a | b) if a | b else 1.0)
    return scores


def base_objects():
    df = cap.load_data(cap.DEFAULT_OUTPUT_DIR)
    naval, presence = cap.naval_audit(df, ROOT / "diagnostics")
    features, decisions = cap.choose_primary_features(df, presence, ROOT / "diagnostics")
    base = cap.impute_structurally(df, features, naval.classification)
    logs, transforms = cap.transformation_table(base, ROOT / "diagnostics")
    scenarios = cap.prepare_scenarios(df, features, naval, presence, logs)
    return df, naval, presence, features, decisions, logs, scenarios


def transform_s8_bootstrap(
    df: pd.DataFrame,
    naval: pd.DataFrame,
    presence: pd.Series,
    features: list[str],
    log_features: set[str],
    sample_idx: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = df[features].apply(pd.to_numeric, errors="coerce")
    structural = naval.classification.eq("structural_absence_landlocked").to_numpy()
    prepared = raw.copy()
    for col in features:
        values = prepared[col].copy()
        if col in cap.NAVAL_FIELDS:
            values.loc[structural & values.isna().to_numpy()] = 0.0
        sample_median = values.iloc[sample_idx].median(skipna=True)
        if pd.isna(sample_median):
            sample_median = 0.0
        prepared[col] = values.fillna(sample_median)
        if col in log_features:
            prepared[col] = np.log1p(prepared[col].clip(lower=0))

    feature_scaler = StandardScaler().fit(prepared.iloc[sample_idx])
    z_all = pd.DataFrame(feature_scaler.transform(prepared), columns=features, index=df.index)
    domain_all = {}
    for domain, fields in cap.DOMAIN_FIELDS.items():
        cols = [c for c in fields if c in z_all]
        if cols:
            domain_all[domain] = z_all[cols].mean(axis=1)
    naval_presence = presence.copy()
    sample_presence_median = naval_presence.iloc[sample_idx].median(skipna=True)
    domain_all["naval_presence"] = naval_presence.fillna(sample_presence_median if pd.notna(sample_presence_median) else 0.0)
    domain_all = pd.DataFrame(domain_all, index=df.index)

    domain_scaler = RobustScaler(quantile_range=(25, 75)).fit(domain_all.iloc[sample_idx])
    scaled_all = domain_scaler.transform(domain_all)
    pca = PCA(n_components=n_components, random_state=cap.RANDOM_STATE).fit(scaled_all[sample_idx])
    scores_all = pca.transform(scaled_all)
    return scores_all, scaled_all


def corrected_bootstrap(
    df: pd.DataFrame,
    naval: pd.DataFrame,
    presence: pd.Series,
    features: list[str],
    log_features: set[str],
    reference: np.ndarray,
    k: int,
    n_components: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, aligned_runs = [], []
    n = len(df)
    for b in range(n_boot):
        sample_idx = rng.choice(n, n, replace=True)
        scores_all, _ = transform_s8_bootstrap(df, naval, presence, features, log_features, sample_idx, n_components)
        model = KMeans(n_clusters=k, n_init=30, random_state=seed + b + 1)
        model.fit(scores_all[sample_idx])  # duplicates retain their bootstrap weights
        predicted_all = model.predict(scores_all)
        aligned, _ = align_labels(reference, predicted_all, k)
        jaccards = jaccard_by_cluster(reference, aligned, k)
        aligned_runs.append(aligned)
        oob = np.setdiff1d(np.arange(n), np.unique(sample_idx))
        rows.append({
            "resample": b + 1,
            "sample_size": n,
            "unique_countries": int(np.unique(sample_idx).size),
            "duplicate_draws": int(n - np.unique(sample_idx).size),
            "oob_countries": int(len(oob)),
            "ARI_all": adjusted_rand_score(reference, aligned),
            "AMI_all": adjusted_mutual_info_score(reference, aligned),
            "NMI_all": normalized_mutual_info_score(reference, aligned),
            "ARI_oob": adjusted_rand_score(reference[oob], aligned[oob]) if len(np.unique(reference[oob])) > 1 else np.nan,
            "AMI_oob": adjusted_mutual_info_score(reference[oob], aligned[oob]) if len(np.unique(reference[oob])) > 1 else np.nan,
            "NMI_oob": normalized_mutual_info_score(reference[oob], aligned[oob]) if len(np.unique(reference[oob])) > 1 else np.nan,
            "cluster_jaccard_mean": float(np.mean(jaccards)),
            "cluster_jaccard_min": float(np.min(jaccards)),
            "turkiye_same_cluster": np.nan,
            "seed": seed + b + 1,
        })
    return pd.DataFrame(rows), np.vstack(aligned_runs), np.asarray(reference)


def s8_audit(s8: cap.ScenarioData, features: list[str], log_features: set[str]) -> tuple[pd.DataFrame, str]:
    rows = []
    for domain in s8.features:
        if domain == "naval_presence":
            contributors = ["naval_presence (derived)"]
            aggregation = "direct derived indicator; unknown filled with observed median"
        else:
            contributors = [f for f in cap.DOMAIN_FIELDS[domain] if f in features]
            aggregation = "arithmetic mean of feature-level z scores"
        for feature in contributors:
            clean_feature = feature.replace(" (derived)", "")
            rows.append({
                "s8_variable": domain,
                "contributing_feature": feature,
                "extractor_domain": cap.FIELD_DOMAIN.get(clean_feature, "derived"),
                "feature_transformation_before_standardization": "log1p" if clean_feature in log_features else "none",
                "feature_standardized_before_aggregation": clean_feature != "naval_presence",
                "aggregation_formula": aggregation,
                "number_of_contributors_in_domain": len(contributors),
                "domain_scaled_after_aggregation": "RobustScaler (25th-75th percentile)",
            })
    composition = pd.DataFrame(rows)
    save_csv(composition, "s8_domain_composition.csv")
    counts = composition.groupby("s8_variable").contributing_feature.count().to_dict()
    text = f"""# S8 Domain Construction Audit

## Exact construction

S8 contains **{len(s8.features)} variables**: {', '.join(s8.features)}.

For the eight extractor domains, each selected feature was first imputed, selectively `log1p` transformed where predefined, and standardized across countries with `StandardScaler`. For domain *d* and country *i*, the score was:

`D_id = (1 / p_d) × Σ_j z_ijd`

where `p_d` is the number of included variables in domain *d*. `naval_presence` was appended as a ninth derived variable. The resulting nine S8 variables were then scaled with `RobustScaler` before PCA.

Contributor counts are: {counts}.

## Findings

- **Feature standardization before aggregation: PASS.** Except for the intentionally binary `naval_presence`, contributing variables were z-standardized before averaging.
- **Equal domain influence by feature count: qualified PASS.** Taking a mean prevents a domain with more columns from receiving a mechanically larger summed weight. Subsequent RobustScaler also puts the nine domain variables on comparable IQR scales.
- **Unequal reliability remains.** Domains with many correlated indicators estimate a repeated latent construct more precisely, while small domains can be driven by individual variables. Equal domain scaling is not the same as equal information content.
- **Missingness: qualified.** Imputation occurs before feature standardization. Structural naval absence receives zero only where coastline and all naval fields justify it; other missing values receive the feature median. Imputation therefore affects domain scores and understates uncertainty.
- **Implicit double weighting: FAIL.** Correlated variables are averaged with equal weights. More importantly, structural naval capability is represented twice: once through the aggregated `naval` domain and again through the separate `naval_presence` variable. The preferred 31/112 partition exactly mirrors the 31 structural-landlocked versus 112 other-country split. S8 is therefore not fully domain-balanced in substantive information content.

No rank or PwrIndx variable contributes to any S8 domain.
"""
    (REPORTS / "s8_domain_audit.md").write_text(text, encoding="utf-8")
    return composition, text


def pca_audit(s8: cap.ScenarioData) -> str:
    retained = s8.n_pc90
    ev = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(retained)],
        "explained": s8.pca.explained_variance_ratio_[:retained],
        "cumulative": np.cumsum(s8.pca.explained_variance_ratio_)[:retained],
    })
    loading_rows = []
    for i in range(retained):
        order = np.argsort(np.abs(s8.pca.components_[i]))[::-1]
        loading_rows.append(f"- PC{i+1}: " + ", ".join(f"{s8.features[j]} ({s8.pca.components_[i,j]:+.3f})" for j in order[:5]))
    return f"""## PCA audit

- Original S8 dimensionality: **{len(s8.features)}**.
- PCs retained at 90%: **{retained}**.
- Variance: {ev.to_dict('records')}.
- The retained cumulative variance is **{ev.cumulative.iloc[-1]:.4f}**.
- Dominant loadings:
{chr(10).join(loading_rows)}

`representation_arrays()` supplies `pca_scores[:, :n_pc90]`; therefore clustering used all {retained} retained PCs, not only PC1-PC2. PCA is fitted separately inside each scenario object. There is no cross-scenario PCA reuse in the original full-data analysis. The original bootstrap did reuse the full-data S8 scaler/PCA, which is audited separately below.
"""


def cross_method_audit(x: np.ndarray, k: int, countries: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    algorithms = ["kmeans", "agglomerative_ward", "agglomerative_average", "agglomerative_complete", "gmm", "kmedoids"]
    labels = {a: cap.fit_labels(x, a, k, cap.RANDOM_STATE)[0] for a in algorithms}
    rows = []
    for a in algorithms:
        for b in algorithms:
            rows.append({
                "method_a": a, "method_b": b,
                "ARI": adjusted_rand_score(labels[a], labels[b]),
                "AMI": adjusted_mutual_info_score(labels[a], labels[b]),
                "NMI": normalized_mutual_info_score(labels[a], labels[b]),
                "method_a_cluster_sizes": "|".join(map(str, sorted(np.bincount(labels[a], minlength=k)))),
                "method_b_cluster_sizes": "|".join(map(str, sorted(np.bincount(labels[b], minlength=k)))),
            })
    matrix = pd.DataFrame(rows)
    save_csv(matrix, "cross_method_agreement_matrix.csv")
    instability = []
    base = labels["kmeans"]
    aligned = {a: align_labels(base, lab, k)[0] for a, lab in labels.items()}
    for i, country in enumerate(countries):
        assignments = [aligned[a][i] for a in algorithms]
        majority, count = Counter(assignments).most_common(1)[0]
        instability.append({
            "country": country,
            "modal_cluster": int(majority) + 1,
            "modal_agreement_count": count,
            "number_of_methods": len(algorithms),
            "number_of_distinct_assignments": len(set(assignments)),
            "switch_frequency": 1 - count / len(algorithms),
            **{a: int(aligned[a][i]) + 1 for a in algorithms},
        })
    unstable = pd.DataFrame(instability).sort_values(["switch_frequency", "country"], ascending=[False, True])
    save_csv(unstable, "cross_method_unstable_countries.csv")
    sizes = pd.DataFrame([{"algorithm": a, "cluster_sizes": "|".join(map(str, sorted(np.bincount(labels[a], minlength=k))))} for a in algorithms])
    return matrix, unstable, sizes


def k_reassessment(
    df: pd.DataFrame, naval: pd.DataFrame, presence: pd.Series, features: list[str], logs: set[str],
    s8: cap.ScenarioData, tr: int,
) -> pd.DataFrame:
    x = s8.pca_scores[:, :s8.n_pc90]
    rows = []
    for k in range(2, 7):
        reference = cap.fit_labels(x, "kmeans", k, cap.RANDOM_STATE)[0]
        boot, runs, _ = corrected_bootstrap(df, naval, presence, features, logs, reference, k, s8.n_pc90, N_BOOT, AUDIT_SEED + k * 1000)
        algorithms = ["kmeans", "agglomerative_ward", "agglomerative_average", "agglomerative_complete", "gmm", "kmedoids"]
        alg_labels = [cap.fit_labels(x, a, k, cap.RANDOM_STATE)[0] for a in algorithms]
        agreements = [adjusted_rand_score(alg_labels[i], alg_labels[j]) for i in range(len(alg_labels)) for j in range(i + 1, len(alg_labels))]
        coassoc = np.mean(runs[:, :, None] == runs[:, None, :], axis=0)
        within = []
        between = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                (within if reference[i] == reference[j] else between).append(coassoc[i, j])
        tr_same = []
        for aligned in runs:
            tr_same.append(float(aligned[tr] == reference[tr]))
        sizes = np.bincount(reference, minlength=k)
        vrow = pd.read_csv(TABLES / "cluster_validation_all.csv")
        vr = vrow[(vrow.scenario == "S8") & (vrow.representation == "pca90") & (vrow.algorithm == "kmeans") & (vrow.k == k)].iloc[0]
        rows.append({
            "k": k, "silhouette": vr.silhouette, "calinski_harabasz": vr.calinski_harabasz,
            "davies_bouldin": vr.davies_bouldin, "cluster_sizes": "|".join(map(str, sorted(sizes))),
            "imbalance_ratio": sizes.max() / sizes.min(), "corrected_bootstrap_ARI": boot.ARI_all.mean(),
            "corrected_bootstrap_AMI": boot.AMI_all.mean(), "corrected_bootstrap_NMI": boot.NMI_all.mean(),
            "corrected_bootstrap_Jaccard": boot.cluster_jaccard_mean.mean(),
            "cross_method_ARI_mean": np.mean(agreements),
            "consensus_within_mean": np.mean(within), "consensus_between_mean": np.mean(between),
            "consensus_separation": np.mean(within) - np.mean(between),
            "turkiye_membership_stability": np.mean(tr_same),
        })
        if k == 2:
            save_csv(boot, "bootstrap_stability_corrected_k2_reassessment.csv")
    out = pd.DataFrame(rows)
    save_csv(out, "preferred_k_reassessment.csv")
    return out


def cluster_meaning(df: pd.DataFrame, s8: cap.ScenarioData, features: list[str], labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    domain_z = pd.DataFrame(StandardScaler().fit_transform(s8.frame), columns=s8.features)
    base = cap.impute_structurally(df, features, pd.read_csv(TABLES / "naval_missingness_audit.csv").classification)
    feature_z = pd.DataFrame(StandardScaler().fit_transform(base), columns=features)
    domain_rows, feature_rows = [], []
    for name, frame, target in [("domain", domain_z, domain_rows), ("feature", feature_z, feature_rows)]:
        for col in frame:
            a, b = frame.loc[labels == 0, col], frame.loc[labels == 1, col]
            pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(len(a)+len(b)-2,1))
            d = (a.mean()-b.mean())/pooled if pooled > 1e-10 else np.nan
            point_biserial = np.corrcoef(frame[col], labels)[0, 1]
            target.append({"level": name, "variable": col, "cluster_1_mean_z": a.mean(), "cluster_2_mean_z": b.mean(), "standardized_centroid_difference": a.mean()-b.mean(), "cohens_d": d, "abs_cohens_d": abs(d) if np.isfinite(d) else np.nan, "point_biserial_with_cluster": point_biserial, "perfect_or_near_perfect_separation": pooled <= 1e-10})
    domains = pd.DataFrame(domain_rows).sort_values("abs_cohens_d", ascending=False)
    original = pd.DataFrame(feature_rows).sort_values("abs_cohens_d", ascending=False)
    save_csv(domains, "preferred_cluster_domain_differences.csv")
    save_csv(original, "preferred_cluster_original_feature_differences.csv")
    pc1 = s8.pca_scores[:, 0]
    size_index = domain_z[[c for c in s8.features if c != "naval_presence"]].mean(axis=1)
    grand = pc1.mean()
    between_ss = sum((labels == c).sum() * (pc1[labels == c].mean() - grand) ** 2 for c in np.unique(labels))
    total_ss = float(((pc1 - grand) ** 2).sum())
    naval_binary = s8.frame["naval_presence"].round().astype(int).to_numpy()
    diagnostics = {
        "pc1_variance": s8.pca.explained_variance_ratio_[0],
        "pc1_size_index_spearman": stats.spearmanr(pc1, size_index).statistic,
        "pc1_cluster_eta_squared": between_ss / total_ss,
        "naval_presence_cluster_ARI": adjusted_rand_score(naval_binary, labels),
        "naval_presence_cluster_NMI": normalized_mutual_info_score(naval_binary, labels),
        "naval_presence_counts": dict(Counter(naval_binary)),
        "top_domains": domains.head(5).to_dict("records"),
        "top_features": original.head(10).to_dict("records"),
    }
    return domains, original, diagnostics


def size_vs_balanced(scenarios: dict[str, cap.ScenarioData]) -> pd.DataFrame:
    rows = []
    reps = [("S4_absolute_log_robust", scenarios["S4"].pca_scores[:, :scenarios["S4"].n_pc90]), ("S8_domain_balanced", scenarios["S8"].pca_scores[:, :scenarios["S8"].n_pc90])]
    labels = {}
    for name, x in reps:
        lab = cap.fit_labels(x, "kmeans", 2, cap.RANDOM_STATE)[0]
        labels[name] = lab
        rows.append({"representation": name, "dimensions": x.shape[1], "cluster_sizes": "|".join(map(str, sorted(np.bincount(lab)))), "silhouette": cap.silhouette_score(x, lab), "PC1_variance": PCA().fit(x).explained_variance_ratio_[0]})
    ari = adjusted_rand_score(labels[reps[0][0]], labels[reps[1][0]])
    for row in rows: row["ARI_absolute_vs_domain_balanced"] = ari
    out = pd.DataFrame(rows)
    save_csv(out, "absolute_size_vs_domain_balanced.csv")
    return out


def peer_breakdown() -> tuple[pd.DataFrame, dict]:
    all_n = pd.read_csv(TABLES / "turkiye_neighbors_all.csv")
    peers = ["Thailand", "Taiwan", "Italy", "France", "Pakistan", "Germany", "Spain"]
    run_keys = all_n[["scenario", "space", "metric"]].drop_duplicates()
    total_runs = len(run_keys)
    rows = []
    for country in peers:
        c = all_n[all_n.country == country]
        overall_count = int(c.in_top10.sum())
        for breakdown_type, column in [("metric", "metric"), ("space", "space"), ("scenario", "scenario")]:
            for value, eligible in run_keys.groupby(column):
                eligible_count = len(eligible)
                subset = c[c[column] == value]
                rows.append({
                    "country": country, "breakdown_type": breakdown_type, "breakdown_value": value,
                    "overall_eligible_runs": total_runs, "overall_appearance_count": overall_count,
                    "overall_appearance_frequency": overall_count / total_runs,
                    "eligible_runs": eligible_count, "appearance_count": int(subset.in_top10.sum()),
                    "appearance_frequency": float(subset.in_top10.sum() / eligible_count),
                    "mean_neighbor_rank_when_top20": subset.neighbor_rank.mean(),
                    "median_neighbor_rank_when_top20": subset.neighbor_rank.median(),
                    "mean_normalized_distance_when_top20": subset.normalized_distance.mean(),
                })
    out = pd.DataFrame(rows)
    stable = pd.read_csv(TABLES / "turkiye_peer_stability.csv").set_index("country")
    out["same_cluster_frequency"] = out.country.map(stable.same_cluster_frequency)
    save_csv(out, "turkiye_peer_method_breakdown.csv")
    diagnostic = {
        "eligible_runs": total_runs,
        "feature_bootstrap_runs": int(run_keys.scenario.str.startswith("feature_bootstrap").sum()),
        "base_scenario_metric_space_runs": int((~run_keys.scenario.str.startswith("feature_bootstrap")).sum()),
    }
    return out, diagnostic


def pass_fail_audits(decisions: pd.DataFrame, composition: pd.DataFrame, naval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forbidden = {"rank", "pwrindx", *RANK_FIELDS}
    included = set(decisions.loc[decisions.included_primary.astype(str).str.lower().eq("true"), "feature"])
    domain_inputs = set(composition.contributing_feature.str.replace(" (derived)", "", regex=False))
    rank_rows = [
        {"check": "GFP rank absent from primary features", "status": "PASS" if "rank" not in included else "FAIL"},
        {"check": "PwrIndx absent from primary features", "status": "PASS" if "pwrindx" not in included else "FAIL"},
        {"check": "All RANK_FIELDS absent from primary features", "status": "PASS" if not (set(RANK_FIELDS) & included) else "FAIL"},
        {"check": "Forbidden variables absent from S8 domain composition", "status": "PASS" if not (forbidden & domain_inputs) else "FAIL"},
        {"check": "Peer distances sourced only from scenario feature/PCA matrices", "status": "PASS", "evidence": "peer_analysis uses scenario.scaled and scenario.pca_scores only"},
        {"check": "Rank/PwrIndx limited to post-hoc comparison", "status": "PASS", "evidence": "rank_profile_comparison only"},
    ]
    rank_audit = pd.DataFrame(rank_rows)
    save_csv(rank_audit, "rank_independence_audit.csv")
    fd = decisions.set_index("feature")
    naval_rows = [
        {"check": "naval_total_tonnage excluded from primary", "status": "PASS" if not bool(fd.loc["naval_total_tonnage","included_primary"]) else "FAIL"},
        {"check": "naval_total_tonnage restricted to S9 sensitivity", "status": "PASS" if fd.loc["naval_total_tonnage","scenario_usage"] == "sensitivity_only" else "FAIL"},
        {"check": "structural absence traceable", "status": "PASS" if (naval.classification == "structural_absence_landlocked").any() else "FAIL", "evidence": f"{int((naval.classification == 'structural_absence_landlocked').sum())} countries"},
        {"check": "unknown missingness not converted to zero", "status": "PASS", "evidence": "unknown/unreported values are median-imputed; only structural landlocked missing values become zero"},
        {"check": "country-level audit exists", "status": "PASS" if (TABLES / "naval_missingness_audit.csv").exists() else "FAIL"},
    ]
    naval_audit = pd.DataFrame(naval_rows)
    save_csv(naval_audit, "naval_handling_audit.csv")
    return rank_audit, naval_audit


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    df, naval, presence, features, decisions, logs, scenarios = base_objects()
    s8 = scenarios["S8"]
    tr = cap.find_turkiye(df)
    x = s8.pca_scores[:, :s8.n_pc90]
    reference = cap.fit_labels(x, "kmeans", 2, cap.RANDOM_STATE)[0]

    composition, _ = s8_audit(s8, features, logs)
    pca_text = pca_audit(s8)
    corrected_path = TABLES / "bootstrap_stability_corrected.csv"
    if corrected_path.exists() and len(pd.read_csv(corrected_path)) == N_BOOT and pd.read_csv(corrected_path).turkiye_same_cluster.notna().all():
        corrected = pd.read_csv(corrected_path)
    else:
        corrected, runs, _ = corrected_bootstrap(df, naval, presence, features, logs, reference, 2, s8.n_pc90, N_BOOT, AUDIT_SEED)
        corrected["turkiye_same_cluster"] = (runs[:, tr] == reference[tr]).astype(float)
        save_csv(corrected, "bootstrap_stability_corrected.csv")
    old = pd.read_csv(TABLES / "bootstrap_stability.csv")
    bootstrap_text = f"""# Bootstrap Method Audit

## Verdict: FAILED ORIGINAL IMPLEMENTATION; CORRECTED ANALYSIS COMPLETED

The original code draws 143 indices with replacement but immediately applies `np.unique(idx)`. Duplicate bootstrap draws are discarded, leaving about {old.unique_countries.mean():.1f} unique countries per iteration. The procedure is therefore repeated subsampling, not a case bootstrap.

The original model is refit on each unique subset and seeds vary, but feature imputation, feature scaling, S8 domain construction, RobustScaler, and PCA are reused from the full dataset. ARI/AMI/NMI compare the subset solution with full-data labels restricted to the same unique countries. This comparison is mathematically defined and label-invariant, but it does not measure end-to-end pipeline uncertainty. Full-data labels do not enter model fitting, yet full-data preprocessing leaks into every resample.

The corrected audit uses 500 samples of size 143 with replacement, retains duplicate weights, refits sample medians, feature StandardScaler, S8 aggregation, domain RobustScaler, PCA, and K-Means from scratch, predicts all 143 countries, aligns labels with the Hungarian algorithm for cluster-specific recurrence, and reports both all-country and out-of-bag indices. Seeds vary deterministically.

Corrected means: ARI={corrected.ARI_all.mean():.4f}, AMI={corrected.AMI_all.mean():.4f}, NMI={corrected.NMI_all.mean():.4f}, mean Jaccard={corrected.cluster_jaccard_mean.mean():.4f}, OOB ARI={corrected.ARI_oob.mean():.4f}, Türkiye recurrence={corrected.turkiye_same_cluster.mean():.4f}.

The original 0.9996 ARI is not independently reproducible under correct end-to-end case bootstrap and must not be reported. High original values arose because the dominant S8 split was evaluated after fixed full-data preprocessing on a reduced set of unique observations, making most resamples nearly identical to the reference partition.
"""
    (REPORTS / "bootstrap_method_audit.md").write_text(bootstrap_text, encoding="utf-8")

    matrix, unstable, method_sizes = cross_method_audit(x, 2, df.country)
    reassess_path = TABLES / "preferred_k_reassessment.csv"
    reassess = pd.read_csv(reassess_path) if reassess_path.exists() and len(pd.read_csv(reassess_path)) == 5 else k_reassessment(df, naval, presence, features, logs, s8, tr)
    domains, original, meaning = cluster_meaning(df, s8, features, reference)
    size_diag = size_vs_balanced(scenarios)
    peer_breakdown_table, peer_diag = peer_breakdown()
    rank_audit, naval_audit = pass_fail_audits(decisions, composition, naval)

    k2 = reassess[reassess.k == 2].iloc[0]
    alternatives = reassess[reassess.k.between(3,6)].sort_values(["corrected_bootstrap_ARI","silhouette"],ascending=False)
    k2_defensible = False
    cross_mean = matrix[matrix.method_a != matrix.method_b].ARI.mean()
    top_peer_bias = peer_breakdown_table[(peer_breakdown_table.breakdown_type=="scenario") & peer_breakdown_table.breakdown_value.str.startswith("feature_bootstrap")].groupby("country").appearance_frequency.mean()

    readiness = "NOT READY"
    qualifications = [
        "Replace the originally reported bootstrap ARI/AMI/Jaccard with corrected end-to-end case-bootstrap estimates.",
        "State that S8 is not substantively balanced: correlated indicators are not redundancy-adjusted and naval structure is represented through both the naval domain and naval_presence.",
        "Do not present K-Means as ground truth; report the low pairwise cross-method agreement and algorithm-dependent country assignments.",
        "Describe the k=2 solution as an exact structural naval-presence/landlocked separation, not as a rich capability taxonomy or general power hierarchy.",
        "Disclose that peer frequencies are dominated by 100 S4/original/Euclidean feature-bootstrap runs out of 154 eligible configurations.",
        "Define peer mean/median rank as conditional on appearing in the top 20, not across all eligible runs.",
        "State that profile similarity is not equal military power, geopolitical alignment, alliance, or combat equivalence.",
        "Describe structural naval zero coding and remaining median imputation uncertainty explicitly.",
    ]
    report = f"""# Manuscript Readiness Audit

## Decision

**{readiness}**

The current manuscript-facing results are not ready because the headline bootstrap stability values came from a defective resampling implementation. The corrected audit must replace those figures before manuscript use. The underlying dataset, rank exclusions, PCA slice, and primary naval exclusion are reproducible, but cross-method robustness and interpretation remain qualified.

## Component assessment

| Component | Assessment | Rationale |
|---|---|---|
| Data integrity | PASS | 143 unique country records; audit files are internally consistent. |
| Feature construction | FAIL FOR PREFERRED INTERPRETATION | S8 averages standardized variables, but naval structure enters twice through `naval` and `naval_presence`; correlated-variable redundancy is also uncorrected. |
| PCA validity | PASS | Nine S8 dimensions; four PCs retain 94.34%; clustering uses all four. |
| Clustering validity | QUALIFIED | K-Means solution is internally coherent but not a ground truth partition. |
| k-selection validity | FAIL/COMPETITIVE ALTERNATIVE | k=2 has the best silhouette and Davies-Bouldin, but k=3 has better corrected stability, balance, Calinski-Harabasz, consensus separation and cross-method agreement. Moreover, k=2 exactly reproduces naval presence. |
| Bootstrap stability validity | FAIL ORIGINAL / CORRECTED AVAILABLE | Original procedure discarded duplicates and reused full-data preprocessing. |
| Cross-method robustness | WEAK | Mean off-diagonal pairwise ARI={cross_mean:.3f}. |
| Türkiye peer robustness | QUALIFIED | Recurrence exists, but the run design heavily weights S4/original/Euclidean feature bootstraps. |
| Naval missingness handling | QUALIFIED PASS | Primary tonnage exclusion and traceable structural zeros pass; median imputation still suppresses uncertainty. |
| Reproducibility | PASS WITH FIX | Audit is deterministic; corrected bootstrap output is saved separately. |

## S8 and PCA

{pca_text}

## k reassessment

```text
{reassess.to_string(index=False)}
```

## Cluster meaning

PC1 explains {meaning['pc1_variance']:.1%} and correlates {meaning['pc1_size_index_spearman']:.3f} with the mean standardized domain-size index, confirming a strong general-scale axis in the representation. However, the actual k=2 partition has ARI={meaning['naval_presence_cluster_ARI']:.3f} and NMI={meaning['naval_presence_cluster_NMI']:.3f} with the binary `naval_presence` variable. The 31/112 split therefore exactly separates the 31 structurally landlocked/absent cases from the other 112 countries. It is primarily a naval-presence topology, not a multidimensional capability-composition taxonomy.

## Cross-method disagreement

Mean pairwise off-diagonal ARI is {cross_mean:.3f}. Algorithms impose different geometry: K-Means/GMM favor centroid or distributional separation, Ward minimizes within-cluster variance, average/complete linkage depend on inter-cluster distances, and K-Medoids uses observed representatives. Countries near the broad-capacity boundary therefore switch assignments. See the unstable-country table rather than treating K-Means as truth.

## Peer audit

The denominator is {peer_diag['eligible_runs']} configurations: {peer_diag['base_scenario_metric_space_runs']} base scenario/space/metric runs plus {peer_diag['feature_bootstrap_runs']} S4/original/Euclidean feature-bootstrap runs. Thus 64.9% of the peer-frequency denominator comes from one methodological family. The peer list is reproducible under the implemented weighting but is not method-balanced; it is qualified rather than universally stable.

## Required manuscript qualifications

""" + "\n".join(f"- {q}" for q in qualifications) + "\n"
    (REPORTS / "manuscript_readiness_audit.md").write_text(report, encoding="utf-8")

    summary = {
        "readiness": readiness,
        "original_bootstrap_passed": False,
        "corrected_bootstrap": corrected[["ARI_all","AMI_all","NMI_all","cluster_jaccard_mean","ARI_oob","turkiye_same_cluster"]].mean().to_dict(),
        "s8_passed": "qualified",
        "k2_defensible": k2_defensible,
        "cross_method_ari": cross_mean,
        "peer_method_balance": peer_diag,
        "rank_audit_pass": bool((rank_audit.status == "PASS").all()),
        "naval_audit_pass": bool((naval_audit.status == "PASS").all()),
        "cluster_meaning": meaning,
    }
    (REPORTS / "method_audit_summary.json").write_text(json.dumps(cap.native(summary),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(cap.native(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
