from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

import audit_capability_profile_analysis as audit
import correct_capability_profile_analysis as corr
import create_capability_profile_analysis as cap


ROOT = Path("outputs/capability_profiles/major_revision")
ROOT.mkdir(parents=True, exist_ok=True)
SEED = corr.SEED


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(ROOT / name, index=False, encoding="utf-8-sig")


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(x) for x in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def save_json(value, name: str) -> None:
    (ROOT / name).write_text(json.dumps(native(value), ensure_ascii=False, indent=2), encoding="utf-8")


def unit_for(field: str) -> str:
    if field in {"purchasing_power", "foreign_reserves", "defense_budget", "external_debt"}:
        return "current USD"
    if field == "square_land_area":
        return "square kilometres"
    if field in {"coastline", "shared_borders", "waterways", "roadway_coverage", "railway_coverage"}:
        return "kilometres"
    if field == "internet_coverage":
        return "percent of population"
    if field in cap.MANPOWER_FIELDS or field in {"labor_force"}:
        return "persons"
    if field in cap.AIRPOWER_FIELDS or field in cap.LAND_FIELDS or field in cap.NAVAL_FIELDS:
        return "asset count"
    if field in {"oil_production", "oil_consumption", "oil_deficit"}:
        return "barrels per day"
    if field == "oil_proven_reserves":
        return "barrels"
    if field in {"natural_gas_production", "natural_gas_consumption", "natural_gas_deficit", "natural_gas_proven_reserves"}:
        return "cubic metres"
    if field in {"coal_production", "coal_consumption", "coal_deficit", "coal_proven_reserves"}:
        return "metric tonnes"
    if field in {"merchant_marine_fleet", "ports_trade_terminals", "airports"}:
        return "count"
    return "count or source-native unit"


def country_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    official_missing = [
        {
            "official_rank": 88,
            "country_official": "Lithuania",
            "present_in_pdf_folder": False,
            "present_in_raw_extraction": False,
            "present_in_clean_dataset": False,
            "exclusion_stage": "source acquisition / raw PDF corpus",
            "reason": "No country PDF was present in gfp_pdfs; therefore no row entered extraction. Not removed by missingness or modeling filters.",
        },
        {
            "official_rank": 114,
            "country_official": "North Macedonia",
            "present_in_pdf_folder": False,
            "present_in_raw_extraction": False,
            "present_in_clean_dataset": False,
            "exclusion_stage": "source acquisition / raw PDF corpus",
            "reason": "No country PDF was present in gfp_pdfs; therefore no row entered extraction. Not removed by missingness or modeling filters.",
        },
    ]
    out = pd.DataFrame(official_missing)
    save_csv(out, "task1_excluded_countries.csv")
    rank_issues = pd.DataFrame([
        {"country": "Benin", "extracted_rank": int(df.loc[df.country.eq("Benin"), "rank"].iloc[0]), "official_rank": 138, "issue": "rank parse/OCR shift; country retained"},
        {"country": "Kosovo", "extracted_rank": int(df.loc[df.country.eq("Kosovo"), "rank"].iloc[0]), "official_rank": 139, "issue": "rank parse/OCR shift; country retained"},
    ])
    save_csv(rank_issues, "task1_rank_extraction_issues.csv")
    return out


def data_dictionary(df, features, decisions, logs) -> pd.DataFrame:
    dec = decisions.set_index("feature")
    rows = []
    for field in features:
        s = pd.to_numeric(df[field], errors="coerce")
        rows.append({
            "variable": field,
            "unit": unit_for(field),
            "domain": cap.FIELD_DOMAIN[field],
            "included_primary": True,
            "pre_domain_transformation": "log1p" if field in logs else "none",
            "feature_standardization": "StandardScaler within bootstrap/full fit",
            "domain_level_scaling": "RobustScaler (IQR) before PCA",
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "unique_nonmissing": int(s.nunique(dropna=True)),
            "structural_naval_rule": "landlocked structural missing -> 0; otherwise median" if field in cap.NAVAL_FIELDS else "median imputation",
            "selection_reason": dec.loc[field, "exclusion_reason"] if field in dec.index else "included",
        })
    out = pd.DataFrame(rows)
    save_csv(out, "task1_variable_dictionary_76.csv")
    return out


def fit_representation(domains: pd.DataFrame, scaling: str):
    scaler = RobustScaler(quantile_range=(25, 75)) if scaling == "robust" else StandardScaler()
    scaled = scaler.fit_transform(domains)
    pca = PCA(random_state=cap.RANDOM_STATE).fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    return scaled, pca, pca.transform(scaled)[:, :n90], n90, cumulative[n90 - 1]


def cluster_grid(name: str, scores: np.ndarray, original_k3: np.ndarray) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows, labels = [], {}
    for k in range(2, 7):
        lab = KMeans(k, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(scores)
        labels[k] = lab
        rows.append({
            "representation": name,
            "k": k,
            "retained_dimensions": scores.shape[1],
            "cluster_sizes": "|".join(map(str, sorted(np.bincount(lab)))),
            "silhouette": silhouette_score(scores, lab),
            "calinski_harabasz": calinski_harabasz_score(scores, lab),
            "davies_bouldin": davies_bouldin_score(scores, lab),
            "ARI_vs_original_k3": adjusted_rand_score(original_k3, lab) if k == 3 else np.nan,
            "AMI_vs_original_k3": adjusted_mutual_info_score(original_k3, lab) if k == 3 else np.nan,
            "NMI_vs_original_k3": normalized_mutual_info_score(original_k3, lab) if k == 3 else np.nan,
        })
    return pd.DataFrame(rows), labels


def peer_top(df, x: np.ndarray, tr: int, representation: str, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for metric in ("euclidean", "mahalanobis", "cosine"):
        distances = cap.pairwise_distances(x, tr, metric)
        order = np.argsort(distances)
        order = order[order != tr][:top_n]
        for rank, idx in enumerate(order, 1):
            rows.append({
                "representation": representation,
                "metric": metric,
                "rank": rank,
                "country": df.country.iloc[idx],
                "distance": distances[idx],
            })
    out = pd.DataFrame(rows)
    return out


def peer_retention(peer_rows: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    n_metrics = peer_rows.metric.nunique()
    return pd.DataFrame([
        {
            "country": country,
            "metrics_in_top10": int(peer_rows.loc[peer_rows.country.eq(country), "metric"].nunique()),
            "retention_rate_across_3_metrics": float(peer_rows.loc[peer_rows.country.eq(country), "metric"].nunique() / n_metrics),
            "mean_rank_when_present": peer_rows.loc[peer_rows.country.eq(country), "rank"].mean(),
        }
        for country in targets
    ])


def mahalanobis_outlier_comparison(domains: pd.DataFrame) -> pd.DataFrame:
    raw = domains.to_numpy()
    robust = RobustScaler(quantile_range=(25, 75)).fit_transform(domains)
    cutoff = chi2.ppf(0.99, df=domains.shape[1])
    rows = []
    for label, matrix in (("before_RobustScaler", raw), ("after_RobustScaler", robust)):
        model = EmpiricalCovariance().fit(matrix)
        md2 = model.mahalanobis(matrix)
        for i, value in enumerate(md2):
            rows.append({"stage": label, "row_index": i, "mahalanobis_d2": value, "chi2_99_cutoff_df8": cutoff, "flagged_99pct": value > cutoff})
    out = pd.DataFrame(rows)
    return out


def ranking_and_baselines(df, domains, corrected_scaled, corrected_scores, n90, tr):
    rank_value = float(df.loc[tr, "rank"])
    rank_neigh = df.loc[df.index != tr, ["country", "rank", "pwrindx"]].copy()
    rank_neigh["absolute_rank_gap"] = (rank_neigh["rank"] - rank_value).abs()
    rank_neigh = rank_neigh.sort_values(["absolute_rank_gap", "rank"]).head(10)
    rank_neigh.insert(0, "neighbor_type", "GFP_rank")

    balanced = pd.read_csv("outputs/capability_profiles/tables/turkiye_peer_balanced.csv")
    profile = balanced.sort_values(["family_presence_frequency", "mean_family_rank"], ascending=[False, True]).head(10).copy()
    profile.insert(0, "neighbor_type", "full_balanced_pipeline")
    profile = profile[["neighbor_type", "country", "family_presence_frequency", "mean_family_rank", "stability_level"]]

    raw_matrix = domains.to_numpy()
    distance = np.linalg.norm(raw_matrix - raw_matrix[tr], axis=1)
    order = np.argsort(distance)
    order = order[order != tr][:10]
    baseline = pd.DataFrame({
        "neighbor_type": "raw_8_domain_euclidean_no_PCA",
        "country": df.country.iloc[order].to_numpy(),
        "distance": distance[order],
        "rank": np.arange(1, 11),
    })
    save_csv(rank_neigh, "task3_gfp_rank_neighbors_top10.csv")
    save_csv(profile, "task3_full_pipeline_profile_neighbors_top10.csv")
    save_csv(baseline, "task3_simple_8domain_euclidean_top10.csv")

    rank_set, profile_set, base_set = set(rank_neigh.country), set(profile.country), set(baseline.country)
    membership = []
    for country in sorted(rank_set | profile_set | base_set):
        membership.append({
            "country": country,
            "in_GFP_rank_top10": country in rank_set,
            "in_full_pipeline_top10": country in profile_set,
            "in_simple_domain_euclidean_top10": country in base_set,
            "rank_vs_profile_status": "overlap" if country in rank_set & profile_set else ("rank_only" if country in rank_set else "profile_only" if country in profile_set else "baseline_only"),
            "complex_vs_simple_status": "overlap" if country in profile_set & base_set else ("complex_only" if country in profile_set else "simple_only" if country in base_set else "neither"),
        })
    membership = pd.DataFrame(membership)
    save_csv(membership, "task3_neighbor_set_comparison.csv")

    domain_z = pd.DataFrame(StandardScaler().fit_transform(domains), columns=domains.columns, index=df.country)
    turkey = domain_z.loc[df.country.iloc[tr]]
    diff_rows = []
    for country in membership.loc[membership.rank_vs_profile_status.ne("overlap"), "country"]:
        delta = (domain_z.loc[country] - turkey).abs().sort_values(ascending=False)
        for rank, (domain, value) in enumerate(delta.items(), 1):
            diff_rows.append({"country": country, "membership_status": membership.set_index("country").loc[country, "rank_vs_profile_status"], "domain": domain, "absolute_standardized_difference": value, "difference_rank": rank})
    differences = pd.DataFrame(diff_rows)
    save_csv(differences, "task3_nonoverlap_domain_differences.csv")
    return rank_neigh, profile, baseline, membership, differences


def peer_scenario_sensitivity(df, scenarios, corrected_scaled, corrected_scores, n90, tr):
    spaces = {}
    for sid in ["S1", "S2", "S3", "S4", "S6", "S7"]:
        s = scenarios[sid]
        spaces[sid] = {"original": s.scaled, "pca": s.pca_scores[:, :s.n_pc90]}
    spaces["S8_corrected"] = {"original": corrected_scaled, "pca": corrected_scores[:, :n90]}
    rows = []
    for sid, reps in spaces.items():
        family = "S8_corrected_only" if sid == "S8_corrected" else "other_six_scenarios"
        for space, x in reps.items():
            for metric in ["euclidean", "mahalanobis", "cosine"]:
                d = cap.pairwise_distances(x, tr, metric)
                order = np.argsort(d)
                order = order[order != tr][:15]
                for rank, idx in enumerate(order, 1):
                    rows.append({"scenario_family": family, "scenario": sid, "space": space, "metric": metric, "country": df.country.iloc[idx], "neighbor_rank": rank, "distance": d[idx]})
    raw = pd.DataFrame(rows)
    save_csv(raw, "taskC_peer_all_configurations_top15.csv")
    summaries = []
    for top_n in [5, 10, 15]:
        for family, total in [("S8_corrected_only", 6), ("other_six_scenarios", 36), ("all_42", 42)]:
            subset = raw[raw.neighbor_rank.le(top_n)] if family == "all_42" else raw[raw.scenario_family.eq(family) & raw.neighbor_rank.le(top_n)]
            for country in df.country[df.index != tr]:
                c = subset[subset.country.eq(country)]
                summaries.append({
                    "top_n": top_n,
                    "scenario_family": family,
                    "country": country,
                    "configurations_total": total,
                    "configurations_present": len(c),
                    "presence_frequency": len(c) / total,
                    "mean_rank_conditional_on_presence": c.neighbor_rank.mean(),
                    "median_rank_conditional_on_presence": c.neighbor_rank.median(),
                })
    summary = pd.DataFrame(summaries).sort_values(["top_n", "scenario_family", "presence_frequency", "mean_rank_conditional_on_presence"], ascending=[True, True, False, True])
    save_csv(summary, "taskC_peer_topN_sensitivity.csv")
    return raw, summary


def main() -> None:
    df, naval, presence, features, decisions, logs, scenarios = audit.base_objects()
    tr = cap.find_turkiye(df)
    country_reconciliation(df)
    dictionary = data_dictionary(df, features, decisions, logs)

    base = cap.impute_structurally(df, features, naval.classification)
    for column in logs:
        base[column] = np.log1p(base[column].clip(lower=0))
    groups, _ = corr.make_redundancy_groups(base, features)
    all_idx = np.arange(len(df))
    prepared = corr.prepare_raw(df, naval, features, logs, all_idx)
    domains, _ = corr.fit_corrected_domains(prepared, all_idx, groups)
    corrected_scaled, corrected_pca, corrected_scores, n90, cumulative90 = fit_representation(domains, "robust")
    original_k3 = KMeans(3, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(corrected_scores)

    # Exact eta-squared scale diagnostic.
    scale, scale_features = corr.scale_index(df, features, logs, naval)
    eta = corr.eta_squared(scale, original_k3)
    grand = scale.mean()
    ss_total = float(((scale - grand) ** 2).sum())
    ss_between = float(sum((original_k3 == c).sum() * (scale[original_k3 == c].mean() - grand) ** 2 for c in np.unique(original_k3)))
    eta_steps = pd.DataFrame([
        {"quantity": "general_scale_index_definition", "value": np.nan, "detail": f"PC1 of StandardScaler-transformed log1p values for {len(scale_features)} nonnegative magnitude indicators; oriented so loading sum is positive"},
        {"quantity": "grand_mean", "value": grand, "detail": "mean of general scale index over 143 countries"},
        {"quantity": "SS_total", "value": ss_total, "detail": "sum_i (scale_i - grand_mean)^2"},
        {"quantity": "SS_between", "value": ss_between, "detail": "sum_c n_c (cluster_mean_c - grand_mean)^2"},
        {"quantity": "eta_squared", "value": eta, "detail": "SS_between / SS_total"},
    ])
    save_csv(eta_steps, "task2_eta_squared_reconstruction.csv")
    pd.DataFrame({"feature": scale_features}).to_csv(ROOT / "task2_general_scale_index_features.csv", index=False, encoding="utf-8-sig")

    # Naval-deletion and unit-variance alternatives.
    no_naval = domains.drop(columns="naval")
    no_nav_scaled, no_nav_pca, no_nav_scores, no_nav_n90, no_nav_cum = fit_representation(no_naval, "robust")
    z_scaled, z_pca, z_scores, z_n90, z_cum = fit_representation(domains, "standard")
    grid_a, labels_a = cluster_grid("7_domain_no_naval_robust_PCA90", no_nav_scores, original_k3)
    grid_b, labels_b = cluster_grid("8_domain_unit_variance_PCA90", z_scores, original_k3)
    grids = pd.concat([grid_a, grid_b], ignore_index=True)
    grids["cumulative_variance_at_PCA90"] = grids.representation.map({"7_domain_no_naval_robust_PCA90": no_nav_cum, "8_domain_unit_variance_PCA90": z_cum})
    save_csv(grids, "task2_alternative_cluster_comparison_k2_k6.csv")

    peer_a = peer_top(df, no_nav_scores, tr, "7_domain_no_naval_PCA90")
    peer_b = peer_top(df, z_scores, tr, "8_domain_unit_variance_PCA90")
    save_csv(pd.concat([peer_a, peer_b], ignore_index=True), "task2_alternative_turkiye_neighbors_top10.csv")
    retention = pd.concat([
        peer_retention(peer_a, ["Thailand", "Taiwan", "Pakistan"]).assign(representation="7_domain_no_naval_PCA90"),
        peer_retention(peer_b, ["Thailand", "Taiwan", "Pakistan"]).assign(representation="8_domain_unit_variance_PCA90"),
    ], ignore_index=True)
    save_csv(retention, "task2_key_peer_retention.csv")

    # Exact domain variance contribution in the covariance matrix used by PCA.
    covariance = np.cov(corrected_scaled, rowvar=False, ddof=1)
    diagonal = np.diag(covariance)
    variance_contrib = pd.DataFrame({"domain": domains.columns, "variance_in_PCA_input": diagonal, "share_of_total_input_variance": diagonal / diagonal.sum()})
    save_csv(variance_contrib, "task2_domain_variance_contributions.csv")
    z_cov = np.cov(z_scaled, rowvar=False, ddof=1)
    unit_variance = pd.DataFrame({"domain": domains.columns, "variance_in_unit_scaled_input": np.diag(z_cov), "share_of_total_input_variance": np.diag(z_cov) / np.diag(z_cov).sum()})
    save_csv(unit_variance, "task2_unit_variance_contributions.csv")
    loading_comparison = pd.DataFrame({
        "domain": domains.columns,
        "robust_scaled_PC1_eigenvector_coefficient": corrected_pca.components_[0],
        "unit_variance_PC1_eigenvector_coefficient": z_pca.components_[0],
        "absolute_reduction": np.abs(corrected_pca.components_[0]) - np.abs(z_pca.components_[0]),
    })
    save_csv(loading_comparison, "task2_PC1_loading_robust_vs_unit_variance.csv")

    outliers = mahalanobis_outlier_comparison(domains)
    outliers["country"] = outliers.row_index.map(df.country)
    save_csv(outliers, "task2_mahalanobis_before_after_robust_scaler.csv")
    out_summary = outliers.groupby("stage").agg(flagged_countries=("flagged_99pct", "sum"), max_mahalanobis_d2=("mahalanobis_d2", "max"), mean_mahalanobis_d2=("mahalanobis_d2", "mean")).reset_index()
    save_csv(out_summary, "task2_mahalanobis_before_after_summary.csv")

    # k=2 versus the landlocked 31-country k=3 cluster.
    k2 = KMeans(2, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(corrected_scores)
    k2_small = set(np.where(k2 == np.argmin(np.bincount(k2)))[0])
    k3_31_label = int(np.where(np.bincount(original_k3) == 31)[0][0])
    k3_31 = set(np.where(original_k3 == k3_31_label)[0])
    intersection = len(k2_small & k3_31)
    union = len(k2_small | k3_31)
    naval_binary = presence.fillna(presence.median()).round().astype(int).to_numpy()
    k2_evidence = {
        "k2_cluster_sizes": sorted(np.bincount(k2).tolist()),
        "k3_cluster_sizes": sorted(np.bincount(original_k3).tolist()),
        "k2_small_vs_k3_31_jaccard": intersection / union,
        "k2_vs_k3_overall_ARI": adjusted_rand_score(k2, original_k3),
        "k2_vs_naval_presence_ARI": adjusted_rand_score(k2, naval_binary),
        "k3_vs_naval_presence_ARI": adjusted_rand_score(original_k3, naval_binary),
        "same_31_countries": k2_small == k3_31,
        "countries_in_shared_31_group": sorted(df.country.iloc[list(k2_small & k3_31)].tolist()),
    }
    save_json(k2_evidence, "task2_k2_landlocked_overlap.json")

    # Ranking and simple baseline comparisons.
    rank_neigh, profile_neigh, baseline, membership, differences = ranking_and_baselines(df, domains, corrected_scaled, corrected_scores, n90, tr)

    # Correct bootstrap extensions for k=3.
    boot, runs = corr.bootstrap_corrected(df, naval, features, logs, groups, original_k3, 3, n90, tr, n_boot=500)
    save_csv(boot, "taskA_k3_bootstrap_all_iterations.csv")
    jaccard_rows = []
    for b, aligned in enumerate(runs, 1):
        scores = audit.jaccard_by_cluster(original_k3, aligned, 3)
        for cluster, score in enumerate(scores):
            jaccard_rows.append({"resample": b, "reference_cluster": cluster + 1, "reference_cluster_size": int((original_k3 == cluster).sum()), "jaccard": score})
    jaccards = pd.DataFrame(jaccard_rows)
    save_csv(jaccards, "taskA_k3_clusterwise_jaccard_500.csv")
    jaccard_summary = jaccards.groupby(["reference_cluster", "reference_cluster_size"]).jaccard.agg(["mean", "std", "median", "min", "max"]).reset_index()
    save_csv(jaccard_summary, "taskA_k3_clusterwise_jaccard_summary.csv")
    stability = pd.DataFrame({"country": df.country, "reference_cluster": original_k3 + 1, "membership_stability": (runs == original_k3[None, :]).mean(axis=0)})
    stability["switch_rate"] = 1 - stability.membership_stability
    stability = stability.sort_values(["membership_stability", "country"])
    save_csv(stability, "taskA_k3_country_membership_stability.csv")
    save_csv(stability.head(20), "taskA_k3_most_unstable_countries_top20.csv")
    pairwise = pd.read_csv("outputs/capability_profiles/tables/s8_corrected_cross_method_agreement.csv")
    pairwise = pairwise[pairwise.k.eq(3)].copy()
    save_csv(pairwise, "taskA_k3_six_method_pairwise_agreement_15pairs.csv")
    oob_summary = pd.DataFrame([{
        "metric": "ARI_oob", "n": int(boot.ARI_oob.notna().sum()), "mean": boot.ARI_oob.mean(), "sd": boot.ARI_oob.std(ddof=1),
        "q025": boot.ARI_oob.quantile(.025), "median": boot.ARI_oob.median(), "q975": boot.ARI_oob.quantile(.975), "min": boot.ARI_oob.min(), "max": boot.ARI_oob.max(),
    }])
    save_csv(oob_summary, "taskA_k3_out_of_bag_ARI_summary.csv")

    # Peer scenarios and top-N sensitivity.
    peer_raw, peer_summary = peer_scenario_sensitivity(df, scenarios, corrected_scaled, corrected_pca.transform(corrected_scaled), n90, tr)

    # PCA reproducibility audit.
    loading_sq = (corrected_pca.components_ ** 2).sum(axis=1)
    pca_audit = {
        "reverse_indicator_rule": "No blanket sign reversal is applied. Rank and PwrIndx fields are excluded. Deficit variables retain their source sign (negative deficits lower the natural-resources score).",
        "within_group_PC1_sign_rule": "Multiply PC1 by +1 when the sum of its coefficients is nonnegative, otherwise by -1.",
        "correlation_group_rule": "Undirected connected components at |Spearman rho| >= 0.90; transitive/chained links merge into one group.",
        "bootstrap_component_count": f"Fixed at {n90}, selected once from the full-sample PCA90 solution; PCA is refitted with n_components={n90} in every bootstrap iteration, but the count is not reselected.",
        "reported_table2_coefficients": "PCA eigenvector coefficients (sklearn pca.components_), not variable-component correlations.",
        "sum_squared_coefficients_by_PC": {f"PC{i+1}": float(v) for i, v in enumerate(loading_sq)},
        "orthonormality_max_abs_error": float(np.max(np.abs(corrected_pca.components_ @ corrected_pca.components_.T - np.eye(len(domains.columns))))),
    }
    save_json(pca_audit, "taskD_pca_reproducibility_audit.json")

    # Silhouette discrepancy and reference audit.
    silhouette_audit = {
        "canonical_k3_silhouette": float(silhouette_score(corrected_scores, original_k3)),
        "threshold_table_value": float(pd.read_csv("outputs/capability_profiles/tables/spearman_threshold_sensitivity.csv").query("spearman_threshold == 0.90").k3_silhouette.iloc[0]),
        "corrected_k_comparison_value": float(pd.read_csv("outputs/capability_profiles/tables/corrected_k_comparison.csv").query("k == 3").silhouette.iloc[0]),
        "explanation": "Both reproducible outputs equal 0.460919. A manuscript value of 0.4602 is a stale transcription/rounding error, not a seed, subset, or preprocessing difference; replace it with 0.4609.",
    }
    save_json(silhouette_audit, "taskE_silhouette_discrepancy_audit.json")
    references = pd.DataFrame([
        {"reference": "MacKinnon & Webb (2017)", "current_use": "bootstrap/cluster-size support", "assessment": "inappropriate for unsupervised cluster-discovery stability; concerns wild cluster bootstrap inference for regression errors", "action": "replace for resampling-stability claim", "recommended_reference": "Hennig (2007), Cluster-wise assessment of cluster stability, doi:10.1016/j.csda.2006.11.025"},
        {"reference": "Kim, Lee & Kang (2018)", "current_use": "multiple internal validity indices", "assessment": "appropriate for multi-index cluster validation, but not evidence for bootstrap stability", "action": "retain only with internal-validity statement", "recommended_reference": "doi:10.1016/j.asoc.2017.11.052"},
        {"reference": "Hardin & Rocke (2005)", "current_use": "Mahalanobis/robust-distance outlier diagnostics", "assessment": "methodologically appropriate for robust-distance diagnostics", "action": "retain; avoid implying that RobustScaler itself removes outliers", "recommended_reference": "doi:10.1198/106186005X77685"},
    ])
    save_csv(references, "taskF_reference_method_fit_audit.csv")

    summary = {
        "task1": {"official_country_count": 145, "analysis_country_count": len(df), "missing_countries": ["Lithuania", "North Macedonia"], "variable_dictionary_rows": len(dictionary)},
        "task2": {
            "eta_squared": eta,
            "naval_variance_share_robust_PCA_input": float(variance_contrib.set_index("domain").loc["naval", "share_of_total_input_variance"]),
            "naval_variance_share_unit_variance_input": float(unit_variance.set_index("domain").loc["naval", "share_of_total_input_variance"]),
            "naval_PC1_coefficient_robust_scaled": float(corrected_pca.components_[0, domains.columns.get_loc("naval")]),
            "naval_PC1_coefficient_unit_variance": float(z_pca.components_[0, domains.columns.get_loc("naval")]),
            "no_naval_k3": grids.query("representation == '7_domain_no_naval_robust_PCA90' and k == 3").iloc[0].to_dict(),
            "unit_variance_k3": grids.query("representation == '8_domain_unit_variance_PCA90' and k == 3").iloc[0].to_dict(),
            "k2_overlap": k2_evidence,
            "outlier_summary": out_summary.to_dict("records"),
            "key_peer_retention": retention.to_dict("records"),
        },
        "task3": {
            "rank_top10": rank_neigh.country.tolist(),
            "profile_top10": profile_neigh.country.tolist(),
            "simple_baseline_top10": baseline.country.tolist(),
            "rank_profile_overlap_count": int(len(set(rank_neigh.country) & set(profile_neigh.country))),
            "complex_simple_overlap_count": int(len(set(profile_neigh.country) & set(baseline.country))),
        },
        "taskA": {"clusterwise_jaccard": jaccard_summary.to_dict("records"), "oob_ARI": oob_summary.to_dict("records"), "most_unstable": stability.head(10).to_dict("records")},
        "taskB": {"label_matching": "Hungarian algorithm via scipy.optimize.linear_sum_assignment on the negative contingency matrix", "turkiye_membership_stability": float((runs[:, tr] == original_k3[tr]).mean())},
        "taskC": {"mean_rank_definition": "conditional on entering the selected top-N list; absent configurations do not enter the mean"},
        "taskD": pca_audit,
        "taskE": silhouette_audit,
    }
    save_json(summary, "major_revision_results.json")
    print(json.dumps(native(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
