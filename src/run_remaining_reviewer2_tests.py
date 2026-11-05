from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, hypergeom, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

import audit_capability_profile_analysis as audit
import correct_capability_profile_analysis as corr
import create_capability_profile_analysis as cap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/capability_profiles/major_revision"
TABLES = ROOT / "outputs/capability_profiles/tables"
RAW = ROOT / "outputs/tables/gfp_2026_clean.csv"
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def save_json(value: object, name: str) -> None:
    (OUT / name).write_text(json.dumps(cap.native(value), ensure_ascii=False, indent=2), encoding="utf-8")


def align(reference: np.ndarray, candidate: np.ndarray, k: int = 3) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=int)
    for left, right in zip(reference, candidate):
        matrix[int(left), int(right)] += 1
    row, col = linear_sum_assignment(-matrix)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([mapping[int(x)] for x in candidate], dtype=int)


def make_groups(base: pd.DataFrame, features: list[str], threshold: float) -> dict[str, list[list[str]]]:
    groups: dict[str, list[list[str]]] = {}
    for domain in corr.DOMAIN_ORDER:
        columns = [x for x in cap.DOMAIN_FIELDS[domain] if x in features]
        groups[domain] = corr.connected_groups(base[columns].corr(method="spearman"), threshold=threshold)
    return groups


def imputed_frame(df: pd.DataFrame, naval: pd.DataFrame, features: list[str], method: str) -> pd.DataFrame:
    frame = df[features].apply(pd.to_numeric, errors="coerce").copy()
    structural = naval.classification.eq("structural_absence_landlocked").to_numpy()
    for column in features:
        if column in cap.NAVAL_FIELDS:
            frame.loc[structural & frame[column].isna().to_numpy(), column] = 0.0
        value = frame[column].median(skipna=True) if method == "median" else frame[column].mean(skipna=True)
        frame[column] = frame[column].fillna(0.0 if pd.isna(value) else value)
    return frame


def fit_representation(
    df: pd.DataFrame,
    naval: pd.DataFrame,
    features: list[str],
    log_features: set[str],
    threshold: float = 0.90,
    imputation: str = "median",
) -> dict[str, object]:
    base = imputed_frame(df, naval, features, imputation)
    for column in log_features:
        base[column] = np.log1p(base[column].clip(lower=0))
    groups = make_groups(base, features, threshold)
    idx = np.arange(len(df))
    domains, models = corr.fit_corrected_domains(base, idx, groups)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(domains)
    scaled = scaler.transform(domains)
    pca = PCA(random_state=cap.RANDOM_STATE).fit(scaled)
    scores = pca.transform(scaled)
    n90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .90) + 1)
    labels = KMeans(3, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(scores[:, :n90])
    return {"base": base, "groups": groups, "components": sum(len(v) for v in groups.values()), "domains": domains, "scaled": scaled, "pca": pca, "scores": scores, "n90": n90, "labels": labels}


def naval_spotcheck(df: pd.DataFrame, naval: pd.DataFrame, presence: pd.Series):
    rows = []
    for country in ["Bolivia", "Paraguay", "Uzbekistan"]:
        idx = int(df.index[df.country.eq(country)][0])
        values = {field: pd.to_numeric(df.loc[idx, field], errors="coerce") for field in cap.NAVAL_FIELDS}
        positive = {key: float(value) for key, value in values.items() if pd.notna(value) and value > 0}
        rows.append({"country": country, "coastline": pd.to_numeric(df.loc[idx, "coastline"], errors="coerce"), "naval_presence": presence.iloc[idx], "classification": naval.classification.iloc[idx], "positive_naval_fields": json.dumps(positive, sort_keys=True), "all_naval_fields": json.dumps({key: None if pd.isna(value) else float(value) for key, value in values.items()}, sort_keys=True), "coding_reason": "any naval field > 0 => 1; otherwise coastline=0 and all naval fields absent/zero => 0"})
    mismatch = pd.DataFrame({"country": df.country, "coastline": pd.to_numeric(df.coastline, errors="coerce"), "naval_presence": presence, "classification": naval.classification})
    mismatch["coastline_present"] = mismatch.coastline.gt(0).astype(int)
    mismatch = mismatch[mismatch.naval_presence.notna() & mismatch.coastline_present.ne(mismatch.naval_presence.astype(int))].copy()
    mismatch["direction"] = np.where(mismatch.coastline_present.eq(1), "coastline>0_but_naval_presence=0", "coastline=0_but_naval_presence=1")
    return pd.DataFrame(rows), mismatch


def alternative_taiwan_and_distances(df, primary):
    alt = pd.read_csv(OUT / "task2_alternative_turkiye_neighbors_top10.csv")
    rows = []
    for representation in ["8_domain_unit_variance_PCA90", "7_domain_no_naval_PCA90"]:
        subset = alt[alt.representation.eq(representation)]
        present = int(subset.country.eq("Taiwan").sum())
        total = int(subset.metric.nunique())
        rows.append({"representation": representation, "top10_occurrences": present, "configurations_total": total, "presence_frequency": present / total})
    tr = cap.find_turkiye(df)
    targets = ["Egypt", "Ukraine", "Thailand", "Iraq", "Pakistan", "Poland", "China", "Spain"]
    distances = []
    for country in targets:
        idx = int(df.index[df.country.eq(country)][0])
        distances.append({"configuration": "S8_corrected_domain_space_euclidean", "country": country, "distance_from_Turkiye": float(np.linalg.norm(primary["scaled"][idx] - primary["scaled"][tr])), "rank_among_142": int(np.where(np.argsort(np.linalg.norm(primary["scaled"] - primary["scaled"][tr], axis=1)) == idx)[0][0])})
    peers = pd.read_csv(TABLES / "turkiye_peer_balanced.csv")
    egypt = peers.loc[peers.country.eq("Egypt")].iloc[0]
    egypt_check = {"occurrences": int(egypt.families_top10), "eligible": int(egypt.families_eligible), "presence_frequency": float(egypt.family_presence_frequency), "mean_rank_conditional_on_presence": float(egypt.mean_family_rank)}
    return pd.DataFrame(rows), pd.DataFrame(distances), egypt_check


def overlap_and_rank_tests(df, primary):
    expected = 10 * 10 / 142
    p_value = float(hypergeom.sf(5, 142, 10, 10))
    pc1 = primary["scores"][:, 0]
    pwr = pd.to_numeric(df.pwrindx, errors="coerce").to_numpy()
    rank = pd.to_numeric(df["rank"], errors="coerce").to_numpy()
    rows = [
        {"test": "null_top10_overlap", "quantity": "expected_overlap", "value": expected},
        {"test": "null_top10_overlap", "quantity": "observed_overlap", "value": 6},
        {"test": "null_top10_overlap", "quantity": "P_X_ge_6_hypergeometric", "value": p_value},
        {"test": "PC1_vs_PwrIndx", "quantity": "pearson_r", "value": pearsonr(pc1, pwr).statistic},
        {"test": "PC1_vs_PwrIndx", "quantity": "spearman_rho", "value": spearmanr(pc1, pwr).statistic},
        {"test": "PC1_vs_rank", "quantity": "pearson_r", "value": pearsonr(pc1, rank).statistic},
        {"test": "PC1_vs_rank", "quantity": "spearman_rho", "value": spearmanr(pc1, rank).statistic},
    ]
    cluster = pd.DataFrame({"cluster": primary["labels"] + 1, "rank": rank}).groupby("cluster").agg(country_count=("rank", "size"), min_rank=("rank", "min"), median_rank=("rank", "median"), max_rank=("rank", "max")).reset_index()
    return pd.DataFrame(rows), cluster


def threshold_sensitivity(df, naval, features, logs, reference):
    rows = []
    for threshold in THRESHOLDS:
        fit = fit_representation(df, naval, features, logs, threshold=threshold)
        rows.append({"spearman_threshold": threshold, "original_variables": len(features), "retained_components": fit["components"], "pca90_components": fit["n90"], "k3_cluster_sizes": "|".join(map(str, sorted(np.bincount(fit["labels"])))), "ari_vs_threshold_0_90": adjusted_rand_score(reference, fit["labels"])})
    return pd.DataFrame(rows)


def leaveout_tests(df, naval, features, logs, reference):
    cases = {"drop_United_States": ["United States"], "drop_China": ["China"], "drop_Iceland": ["Iceland"], "drop_all_three": ["United States", "China", "Iceland"]}
    rows = []
    for case, removed in cases.items():
        keep = ~df.country.isin(removed)
        sub_df = df.loc[keep].reset_index(drop=True)
        sub_naval = naval.loc[keep].reset_index(drop=True)
        fit = fit_representation(sub_df, sub_naval, features, logs)
        restricted = reference[keep.to_numpy()]
        aligned = align(restricted, fit["labels"])
        changed = sub_df.country[aligned != restricted].tolist()
        rows.append({"case": case, "removed_countries": "|".join(removed), "n_countries": len(sub_df), "ari_vs_original_restricted": adjusted_rand_score(restricted, aligned), "cluster_sizes": "|".join(map(str, sorted(np.bincount(fit["labels"])))), "changed_assignment_count_after_alignment": len(changed), "changed_countries": "|".join(changed), "pca90_components": fit["n90"], "retained_components": fit["components"]})
    return pd.DataFrame(rows)


def mahalanobis_spaces(df, primary):
    rows = []
    spaces = {"original_8_domain_space": np.asarray(primary["domains"]), "PCA90_4_component_space": primary["scores"][:, :primary["n90"]]}
    for name, x in spaces.items():
        cutoff = float(chi2.ppf(.99, df=x.shape[1]))
        values = EmpiricalCovariance().fit(x).mahalanobis(x)
        for idx, value in enumerate(values):
            rows.append({"space": name, "dimensions": x.shape[1], "chi2_99_cutoff": cutoff, "country": df.country.iloc[idx], "mahalanobis_d2": value, "flagged_99pct": bool(value > cutoff)})
    return pd.DataFrame(rows)


def s5_overlap(df, scenarios, primary):
    tr = cap.find_turkiye(df)
    rows = []
    s5 = scenarios["S5"]
    comparisons = [("original", s5.scaled, primary["scaled"]), ("pca90", s5.pca_scores[:, :s5.n_pc90], primary["scores"][:, :primary["n90"]])]
    for space, left, right in comparisons:
        left_order = [i for i in np.argsort(np.linalg.norm(left - left[tr], axis=1)) if i != tr][:10]
        right_order = [i for i in np.argsort(np.linalg.norm(right - right[tr], axis=1)) if i != tr][:10]
        left_names, right_names = df.country.iloc[left_order].tolist(), df.country.iloc[right_order].tolist()
        overlap = sorted(set(left_names) & set(right_names))
        rows.append({"space": space, "S5_top10": left_names, "S8_corrected_top10": right_names, "overlap_count": len(overlap), "overlap_countries": overlap, "jaccard": len(overlap) / len(set(left_names) | set(right_names))})
    return rows


def imputation_sensitivity(df, naval, features, logs, reference):
    mean_fit = fit_representation(df, naval, features, logs, imputation="mean")
    aligned = align(reference, mean_fit["labels"])
    changed = df.country[aligned != reference].tolist()
    return {"alternative": "feature mean imputation; structural landlocked naval missing values remain zero", "ari_vs_primary_median_imputation": adjusted_rand_score(reference, aligned), "changed_assignment_count": len(changed), "changed_countries": changed, "cluster_sizes": sorted(np.bincount(mean_fit["labels"]).tolist()), "retained_components": mean_fit["components"], "pca90_components": mean_fit["n90"]}


def membership_and_turkiye(df, primary):
    membership = pd.DataFrame({"country": df.country, "k3_cluster": primary["labels"] + 1})
    boot = pd.read_csv(TABLES / "s8_corrected_bootstrap_stability.csv")
    tr = cap.find_turkiye(df)
    rows = []
    for k in range(2, 7):
        labels = KMeans(k, n_init=30, random_state=cap.RANDOM_STATE).fit_predict(primary["scores"][:, :primary["n90"]])
        rows.append({"k": k, "turkiye_cluster_id": int(labels[tr] + 1), "bootstrap_membership_stability": float(boot.loc[boot.k.eq(k), "turkiye_membership_stability"].mean()), "bootstrap_membership_stability_percent": float(100 * boot.loc[boot.k.eq(k), "turkiye_membership_stability"].mean())})
    return membership, pd.DataFrame(rows)


def source_and_figure_audit(df):
    local = df.set_index("country")
    source_rows = [
        {"country_requested": "Lithuania", "dataset_country": "", "present_in_143": False, "local_rank": np.nan, "local_pwrindx": np.nan, "official_detail_rank": 88, "official_listing_rank_current": 88, "official_pwrindx": 1.8949, "finding": "Absent because no source PDF entered acquisition; adding requires acquisition/extraction and full downstream rerun."},
        {"country_requested": "North Macedonia", "dataset_country": "", "present_in_143": False, "local_rank": np.nan, "local_pwrindx": np.nan, "official_detail_rank": 114, "official_listing_rank_current": 114, "official_pwrindx": 2.4956, "finding": "Absent because no source PDF entered acquisition; adding requires acquisition/extraction and full downstream rerun."},
        {"country_requested": "Benin", "dataset_country": "Benin", "present_in_143": True, "local_rank": local.loc["Benin", "rank"], "local_pwrindx": local.loc["Benin", "pwrindx"], "official_detail_rank": 137, "official_listing_rank_current": 138, "official_pwrindx": 3.8963, "finding": "PwrIndx matches. Local PDF/detail rank matches 137; current official aggregate listing shows 138, an official-source inconsistency rather than OCR corruption."},
        {"country_requested": "Kosovo", "dataset_country": "Kosovo", "present_in_143": True, "local_rank": local.loc["Kosovo", "rank"], "local_pwrindx": local.loc["Kosovo", "pwrindx"], "official_detail_rank": 138, "official_listing_rank_current": 139, "official_pwrindx": 3.8041, "finding": "PwrIndx matches. Local PDF/detail rank matches 138; current official aggregate listing shows 139, an official-source inconsistency rather than OCR corruption."},
        {"country_requested": "Belize", "dataset_country": "Beliz", "present_in_143": True, "local_rank": local.loc["Beliz", "rank"], "local_pwrindx": local.loc["Beliz", "pwrindx"], "official_detail_rank": 144, "official_listing_rank_current": 144, "official_pwrindx": 4.3602, "finding": "No discrepancy. GFP source and dataset both use the spelling 'Beliz', not 'Belize'."},
    ]
    peers = pd.read_csv(TABLES / "turkiye_peer_balanced.csv")
    figure = peers.copy()
    figure["included_current_figure_top15"] = figure.index < 15
    figure["included_moderate_plus_threshold"] = figure.family_presence_frequency.ge(.30)
    return pd.DataFrame(source_rows), figure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, naval, presence, features, decisions, logs, scenarios = audit.base_objects()
    primary = fit_representation(df, naval, features, logs)
    reference = primary["labels"]

    spot, mismatch = naval_spotcheck(df, naval, presence)
    save_csv(spot, "reviewer_r22_naval_presence_spotcheck.csv")
    save_csv(mismatch, "reviewer_r22_coastline_naval_presence_disagreements.csv")

    taiwan, distances, egypt = alternative_taiwan_and_distances(df, primary)
    save_csv(taiwan, "reviewer_r26_taiwan_alternative_recurrence.csv")
    save_csv(distances, "reviewer_r26_s8_domain_euclidean_distances.csv")
    save_json(egypt, "reviewer_r26_egypt_recurrence_check.json")

    overlap, cluster_ranks = overlap_and_rank_tests(df, primary)
    save_csv(overlap, "reviewer_r27_null_overlap_and_pc1_rank.csv")
    save_csv(cluster_ranks, "reviewer_r27_k3_cluster_rank_distribution.csv")

    threshold = threshold_sensitivity(df, naval, features, logs, reference)
    save_csv(threshold, "reviewer_r28_spearman_threshold_070_095.csv")

    leaveout = leaveout_tests(df, naval, features, logs, reference)
    mahal = mahalanobis_spaces(df, primary)
    s5 = s5_overlap(df, scenarios, primary)
    save_csv(leaveout, "reviewer_r211_extreme_leaveout_sensitivity.csv")
    save_csv(mahal, "reviewer_r211_mahalanobis_domain_vs_pca.csv")
    save_json(s5, "reviewer_r211_s5_winsorized_peer_overlap.json")

    imputation = imputation_sensitivity(df, naval, features, logs, reference)
    save_json(imputation, "reviewer_r212_mean_imputation_sensitivity.json")

    membership, turkiye = membership_and_turkiye(df, primary)
    save_csv(membership, "task_cluster_membership_k3.csv")
    save_csv(turkiye, "reviewer_r216_turkiye_k2_k6_stability.csv")

    source, figure = source_and_figure_audit(df)
    save_csv(source, "reviewer_r217_data_source_audit.csv")
    save_csv(figure, "reviewer_r217_figure7_peer_list_audit.csv")

    summary = {
        "r22": {"spotcheck": spot.to_dict("records"), "coastline_presence_disagreement_count": len(mismatch), "disagreements": mismatch.to_dict("records")},
        "r26": {"taiwan_alternatives": taiwan.to_dict("records"), "representative_distances": distances.to_dict("records"), "egypt": egypt},
        "r27": {"statistics": overlap.to_dict("records"), "cluster_rank_distribution": cluster_ranks.to_dict("records")},
        "r28": threshold.to_dict("records"),
        "r211": {"leaveout": leaveout.to_dict("records"), "mahalanobis_flagged": mahal[mahal.flagged_99pct].to_dict("records"), "s5": s5},
        "r212": imputation,
        "r216": turkiye.to_dict("records"),
        "r217": {"source_audit": source.to_dict("records"), "china": figure[figure.country.eq("China")].to_dict("records"), "moderate_plus": figure[figure.included_moderate_plus_threshold].to_dict("records")},
    }
    save_json(summary, "reviewer_remaining_r2_summary.json")
    print(json.dumps(cap.native(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
