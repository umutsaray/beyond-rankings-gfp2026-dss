from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from diptest import diptest
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

import audit_capability_profile_analysis as audit
import correct_capability_profile_analysis as corr
import create_capability_profile_analysis as cap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/capability_profiles/major_revision"
TABLES = ROOT / "outputs/capability_profiles/tables"
SEED = corr.SEED
N_BOOT = 500
METHODS = ["agglomerative_ward", "agglomerative_average", "agglomerative_complete", "gmm", "kmedoids"]


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def save_json(obj: object, name: str) -> None:
    (OUT / name).write_text(json.dumps(cap.native(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def align(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    contingency = np.zeros((k, k), dtype=int)
    for r, c in zip(reference, candidate):
        contingency[int(r), int(c)] += 1
    row, col = linear_sum_assignment(-contingency)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([mapping[int(x)] for x in candidate], dtype=int)


def canonical_objects():
    df, naval, presence, features, decisions, logs, scenarios = audit.base_objects()
    base = cap.impute_structurally(df, features, naval.classification)
    for column in logs:
        base[column] = np.log1p(base[column].clip(lower=0))
    groups, _ = corr.make_redundancy_groups(base, features)
    _, domains, scaled, pca, scores, _, n90, _ = corr.full_corrected_s8(df, naval, features, logs, groups)
    return df, naval, features, logs, groups, domains, scaled, pca, scores[:, :n90], scenarios, n90


def bootstrap_space(df, naval, features, logs, groups, sample: np.ndarray, n_pc: int) -> np.ndarray:
    prepared = corr.prepare_raw(df, naval, features, logs, sample)
    domains, _ = corr.fit_corrected_domains(prepared, sample, groups)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(domains.iloc[sample])
    scaled = scaler.transform(domains)
    pca = PCA(n_components=n_pc, random_state=cap.RANDOM_STATE).fit(scaled[sample])
    return pca.transform(scaled)


def predict_bootstrap_method(x: np.ndarray, sample: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    train = x[sample]
    if method == "kmeans":
        return KMeans(k, n_init=30, random_state=seed).fit(train).predict(x)
    if method == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-5, random_state=seed).fit(train).predict(x)
    if method.startswith("agglomerative_"):
        linkage = method.split("_", 1)[1]
        train_labels = AgglomerativeClustering(k, linkage=linkage).fit_predict(train)
        centers = np.vstack([train[train_labels == cluster].mean(axis=0) for cluster in range(k)])
        return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    if method == "kmedoids":
        _, medoid_indices, _ = cap.kmedoids(train, k, seed=seed)
        medoids = train[medoid_indices]
        return ((x[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    raise ValueError(method)


def gap_test(x: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 901)
    mins, maxs = x.min(axis=0), x.max(axis=0)
    rows = []
    for k in range(2, 7):
        observed = KMeans(k, n_init=30, random_state=cap.RANDOM_STATE).fit(x).inertia_
        null_logs = []
        for b in range(N_BOOT):
            null = rng.uniform(mins, maxs, size=x.shape)
            inertia = KMeans(k, n_init=10, random_state=SEED + k * 1000 + b).fit(null).inertia_
            null_logs.append(np.log(inertia))
        null_logs = np.asarray(null_logs)
        gap = null_logs.mean() - np.log(observed)
        sk = np.sqrt(1 + 1 / N_BOOT) * null_logs.std(ddof=1)
        p = (1 + np.sum(null_logs <= np.log(observed))) / (N_BOOT + 1)
        rows.append({"k": k, "observed_within_dispersion": observed, "mean_null_log_dispersion": null_logs.mean(), "gap": gap, "gap_se": sk, "null_p_value_tighter_or_equal_than_observed": p})
    out = pd.DataFrame(rows)
    out["max_gap_selected"] = out.k.eq(out.loc[out.gap.idxmax(), "k"])
    selected = int(out.iloc[-1].k)
    for i in range(len(out) - 1):
        if out.iloc[i].gap >= out.iloc[i + 1].gap - out.iloc[i + 1].gap_se:
            selected = int(out.iloc[i].k)
            break
    out["tibshirani_1se_selected"] = out.k.eq(selected)
    return out


def dip_tests(x: np.ndarray) -> pd.DataFrame:
    rows = []
    for component in range(2):
        stat, p = diptest(x[:, component], boot_pval=True, n_boot=10000, seed=SEED + component)
        rows.append({"component": f"PC{component + 1}", "n": len(x), "dip_statistic": stat, "bootstrap_p_value": p, "reject_unimodality_at_0_05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def gmm_information(x: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(2, 7):
        model = GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-5, random_state=cap.RANDOM_STATE).fit(x)
        probabilities = np.clip(model.predict_proba(x), 1e-15, 1)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        bic = float(model.bic(x))
        rows.append({"k": k, "bic": bic, "classification_entropy": entropy, "icl_bic_plus_2entropy": bic + 2 * entropy, "cluster_sizes": "|".join(map(str, sorted(np.bincount(model.predict(x), minlength=k))))})
    out = pd.DataFrame(rows)
    out["bic_selected"] = out.k.eq(out.loc[out.bic.idxmin(), "k"])
    out["icl_selected"] = out.k.eq(out.loc[out.icl_bic_plus_2entropy.idxmin(), "k"])
    return out


def run_k3_bootstraps(df, naval, features, logs, groups, x_ref: np.ndarray, n90: int):
    references = {method: cap.fit_labels(x_ref, method, 3, cap.RANDOM_STATE)[0] for method in METHODS}
    kmeans_ref = cap.fit_labels(x_ref, "kmeans", 3, cap.RANDOM_STATE)[0]
    rng = np.random.default_rng(SEED + 3000)
    iteration_rows, method_runs = [], {method: [] for method in ["kmeans", *METHODS]}
    silhouette_rows = []
    for b in range(N_BOOT):
        sample = rng.choice(len(df), len(df), replace=True)
        xb = bootstrap_space(df, naval, features, logs, groups, sample, n90)
        seed = SEED + 3000 + b + 1
        for method in ["kmeans", *METHODS]:
            reference = kmeans_ref if method == "kmeans" else references[method]
            prediction = predict_bootstrap_method(xb, sample, method, 3, seed)
            aligned = align(reference, prediction, 3)
            ari = adjusted_rand_score(reference, aligned)
            method_runs[method].append(aligned)
            iteration_rows.append({"method": method, "iteration": b + 1, "ari_vs_method_full_sample": ari})
        pred2 = predict_bootstrap_method(xb, sample, "kmeans", 2, seed)
        pred3 = method_runs["kmeans"][-1]
        sil2 = silhouette_score(xb, pred2)
        sil3 = silhouette_score(xb, pred3)
        silhouette_rows.append({"iteration": b + 1, "silhouette_k2": sil2, "silhouette_k3": sil3, "difference_k3_minus_k2": sil3 - sil2})
    method_runs = {key: np.asarray(value) for key, value in method_runs.items()}
    return pd.DataFrame(iteration_rows), method_runs, pd.DataFrame(silhouette_rows), references, kmeans_ref


def summarize_method_bootstrap(iterations: pd.DataFrame) -> pd.DataFrame:
    return iterations.groupby("method").ari_vs_method_full_sample.agg(
        n="count", mean="mean", sd="std", median="median", q025=lambda x: x.quantile(.025), q975=lambda x: x.quantile(.975), below_0_6=lambda x: (x < .6).mean()
    ).reset_index()


def consensus_row(k: int, reference: np.ndarray, runs: np.ndarray) -> dict[str, object]:
    same_ref = reference[:, None] == reference[None, :]
    upper = np.triu(np.ones(same_ref.shape, dtype=bool), 1)
    within_mask, between_mask = upper & same_ref, upper & ~same_ref
    separations = []
    aris = []
    for run in runs:
        same_run = run[:, None] == run[None, :]
        separations.append(float(same_run[within_mask].mean() - same_run[between_mask].mean()))
        aris.append(adjusted_rand_score(reference, run))
    pearson = pearsonr(aris, separations)
    spearman = spearmanr(aris, separations)
    return {"k": k, "n_iterations": len(runs), "bootstrap_ari_mean": np.mean(aris), "consensus_separation_from_same_runs": np.mean(separations), "pearson_r_iteration_ari_vs_separation": pearson.statistic, "pearson_p": pearson.pvalue, "spearman_rho_iteration_ari_vs_separation": spearman.statistic, "spearman_p": spearman.pvalue, "independent_evidence": False, "reason": "Both statistics are deterministic summaries of the same aligned bootstrap label matrices."}


def k4_runs(df, naval, features, logs, groups, x_ref, n90):
    reference = cap.fit_labels(x_ref, "kmeans", 4, cap.RANDOM_STATE)[0]
    rng = np.random.default_rng(SEED + 4000)
    runs = []
    for b in range(N_BOOT):
        sample = rng.choice(len(df), len(df), replace=True)
        xb = bootstrap_space(df, naval, features, logs, groups, sample, n90)
        pred = predict_bootstrap_method(xb, sample, "kmeans", 4, SEED + 4000 + b + 1)
        runs.append(align(reference, pred, 4))
    return reference, np.asarray(runs)


def method_split_table(df, naval, x, references, kmeans_ref):
    cluster31 = int(np.where(np.bincount(kmeans_ref) == 31)[0][0])
    fixed = kmeans_ref == cluster31
    rows = []
    all_refs = {"kmeans": kmeans_ref, **references}
    for method, labels in all_refs.items():
        fixed_labels, fixed_counts = np.unique(labels[fixed], return_counts=True)
        remaining_counts = sorted(np.unique(labels[~fixed], return_counts=True)[1].tolist())
        exact = any(set(np.where(labels == label)[0]) == set(np.where(fixed)[0]) for label in np.unique(labels))
        rows.append({"method": method, "full_cluster_sizes": "|".join(map(str, sorted(np.bincount(labels, minlength=3)))), "landlocked31_number_of_pieces": len(fixed_labels), "landlocked31_piece_sizes": "|".join(map(str, sorted(fixed_counts, reverse=True))), "exact_landlocked31_cluster_recovered": exact, "remaining112_restricted_split_sizes": "|".join(map(str, remaining_counts))})
    return pd.DataFrame(rows), fixed


def distribution_and_country_tests(df, naval, kmeans_ref, kmeans_runs, iteration_df, fixed):
    ari = iteration_df.loc[iteration_df.method.eq("kmeans"), "ari_vs_method_full_sample"].to_numpy()
    dip, dip_p = diptest(ari, boot_pval=True, n_boot=10000, seed=SEED + 25)
    grid = np.linspace(0, 1, 2001)
    density = gaussian_kde(ari)(grid)
    raw_peaks, _ = find_peaks(density)
    prominent_peaks, _ = find_peaks(density, prominence=0.05 * density.max())
    summary = {"n": len(ari), "mean": ari.mean(), "median": np.median(ari), "sd": ari.std(ddof=1), "q025": np.quantile(ari, .025), "q975": np.quantile(ari, .975), "iterations_below_0_6": int((ari < .6).sum()), "proportion_below_0_6": float((ari < .6).mean()), "ari_dip_statistic": dip, "ari_dip_bootstrap_p_value": dip_p, "kde_bandwidth_rule": "scipy gaussian_kde Scott default", "kde_raw_local_mode_count": len(raw_peaks), "kde_raw_mode_locations": grid[raw_peaks].tolist(), "kde_prominent_mode_definition": "peak prominence >= 5% of maximum KDE density", "kde_prominent_mode_count": len(prominent_peaks), "kde_prominent_mode_locations": grid[prominent_peaks].tolist()}
    stability = (kmeans_runs == kmeans_ref[None, :]).mean(axis=0)
    country = pd.DataFrame({"country": df.country, "reference_cluster": kmeans_ref + 1, "structural_landlocked31": fixed, "membership_stability": stability, "switch_rate": 1 - stability})
    land = country[fixed].copy()
    groups = land.groupby("membership_stability").country.agg(["count", lambda x: "|".join(sorted(x))]).reset_index()
    groups.columns = ["membership_stability", "country_count", "countries"]
    unstable = country[country.membership_stability < .80]
    cluster_counts = unstable.groupby("reference_cluster").size().rename("unstable_country_count").reset_index()
    cluster_counts["all_unstable_count"] = len(unstable)
    cluster_counts["share_of_all_unstable"] = cluster_counts.unstable_country_count / len(unstable) if len(unstable) else np.nan
    target = np.where(fixed & np.isclose(stability, 0.812))[0]
    switched = kmeans_runs[:, target] != kmeans_ref[target][None, :]
    joint = {
        "target_stability": 0.812,
        "countries": df.country.iloc[target].tolist(),
        "country_count": len(target),
        "all_six_switch_indicators_identical_in_every_iteration": bool(len(target) and np.all(switched == switched[:, [0]])),
        "iterations_all_six_switch": int(switched.all(axis=1).sum()) if len(target) else 0,
        "iterations_none_of_six_switch": int((~switched).all(axis=1).sum()) if len(target) else 0,
        "iterations_partial_switch": int(((switched.any(axis=1)) & (~switched.all(axis=1))).sum()) if len(target) else 0,
    }
    return summary, country, groups, cluster_counts, joint


def peer_tests(df, scenarios):
    raw = pd.read_csv(OUT / "taskC_peer_all_configurations_top15.csv")
    metric_rows = []
    for metric in ["euclidean", "cosine"]:
        subset = raw[(raw.metric == metric) & (raw.neighbor_rank <= 10)]
        total = subset[["scenario", "space"]].drop_duplicates().shape[0]
        assert total == 14
        for country in df.country:
            if country == "Turkiye":
                continue
            c = subset[subset.country == country]
            frequency = len(c) / total
            level = "HIGH" if frequency >= .60 else "MODERATE" if frequency >= .30 else "LOW"
            metric_rows.append({"metric": metric, "country": country, "configurations_total": total, "top10_occurrences": len(c), "top10_presence_frequency": frequency, "mean_rank_conditional_on_presence": c.neighbor_rank.mean(), "metric_specific_level": level, "level_definition": "HIGH >= 0.60; MODERATE >= 0.30; LOW < 0.30. Original cross-metric diversity requirements are inapplicable within a single-metric subset."})
    metric_df = pd.DataFrame(metric_rows)

    sensitivity_rows = []
    s8 = raw[raw.scenario == "S8_corrected"]
    for top_n in [5, 10, 15]:
        subset = s8[s8.neighbor_rank <= top_n]
        for country in df.country:
            if country == "Turkiye":
                continue
            c = subset[subset.country == country]
            sensitivity_rows.append({"top_n": top_n, "country": country, "configurations_total": 6, "configurations_present": len(c), "presence_frequency": len(c) / 6, "mean_rank_conditional_on_presence": c.neighbor_rank.mean(), "median_rank_conditional_on_presence": c.neighbor_rank.median()})
    sensitivity = pd.DataFrame(sensitivity_rows)

    baseline = pd.read_csv(OUT / "task3_simple_8domain_euclidean_top10.csv")
    direct = raw[(raw.scenario == "S8_corrected") & (raw.space == "original") & (raw.metric == "euclidean") & (raw.neighbor_rank <= 10)].sort_values("neighbor_rank")
    a, b = baseline.country.tolist(), direct.country.tolist()
    overlap = [country for country in a if country in set(b)]
    egypt_all = raw[(raw.country == "Egypt") & (raw.neighbor_rank <= 10)]
    baseline_json = {"baseline_top10": a, "s8_corrected_original_euclidean_top10": b, "overlap_count": len(overlap), "overlap_countries": overlap, "jaccard": len(overlap) / len(set(a) | set(b)), "identical_lists": a == b, "egypt_all42_top10_occurrences": len(egypt_all), "egypt_all42_presence_frequency": len(egypt_all) / 42, "egypt_mean_rank_conditional_on_presence": egypt_all.neighbor_rank.mean(), "egypt_mean_rank_denominator": len(egypt_all), "egypt_total_eligible_configurations": 42}
    return metric_df, sensitivity, baseline_json


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, naval, features, logs, groups, domains, scaled, pca, x, scenarios, n90 = canonical_objects()

    gap = gap_test(x)
    dip = dip_tests(x)
    gmm = gmm_information(x)
    save_csv(gap, "reviewer_r24_gap_null_reference.csv")
    save_csv(dip, "reviewer_r24_pc1_pc2_dip_tests.csv")
    save_csv(gmm, "reviewer_r24_gmm_bic_icl.csv")

    iterations, runs, silhouette, references, kmeans_ref = run_k3_bootstraps(df, naval, features, logs, groups, x, n90)
    method_summary = summarize_method_bootstrap(iterations)
    save_csv(iterations, "reviewer_r24_method_bootstrap_500_iterations.csv")
    save_csv(method_summary, "reviewer_r24_method_bootstrap_summary.csv")

    k4_ref, k4_boot = k4_runs(df, naval, features, logs, groups, x, n90)
    dependence = pd.DataFrame([consensus_row(3, kmeans_ref, runs["kmeans"]), consensus_row(4, k4_ref, k4_boot)])
    save_csv(dependence, "reviewer_r24_consensus_bootstrap_dependence.csv")

    observed = {}
    for k in [2, 3]:
        labels = cap.fit_labels(x, "kmeans", k, cap.RANDOM_STATE)[0]
        observed[k] = silhouette_score(x, labels)
    delta = silhouette.difference_k3_minus_k2
    sil_summary = pd.DataFrame([{"n_iterations": len(silhouette), "observed_silhouette_k2": observed[2], "observed_silhouette_k3": observed[3], "observed_difference_k3_minus_k2": observed[3] - observed[2], "bootstrap_mean_difference": delta.mean(), "bootstrap_sd_difference": delta.std(ddof=1), "bootstrap_median_difference": delta.median(), "q025": delta.quantile(.025), "q975": delta.quantile(.975), "proportion_k3_greater_than_k2": (delta > 0).mean(), "one_sided_p_k3_not_greater": (1 + (delta <= 0).sum()) / (len(delta) + 1), "monte_carlo_se_of_proportion": np.sqrt((delta > 0).mean() * (1 - (delta > 0).mean()) / len(delta)), "monte_carlo_se_of_mean_difference": delta.std(ddof=1) / np.sqrt(len(delta))}])
    save_csv(silhouette, "reviewer_r24_silhouette_k3_vs_k2_bootstrap_500.csv")
    save_csv(sil_summary, "reviewer_r24_silhouette_k3_vs_k2_summary.csv")

    split, fixed = method_split_table(df, naval, x, references, kmeans_ref)
    save_csv(split, "reviewer_r24_method_landlocked31_and_remaining112.csv")

    dist_summary, country, land_groups, unstable, land_joint = distribution_and_country_tests(df, naval, kmeans_ref, runs["kmeans"], iterations, fixed)
    save_json(dist_summary, "reviewer_r25_k3_bootstrap_distribution.json")
    save_csv(country, "reviewer_r25_country_membership_stability.csv")
    save_csv(land_groups, "reviewer_r25_landlocked_stability_groups.csv")
    save_csv(unstable, "reviewer_r25_unstable_below080_cluster_composition.csv")
    save_json(land_joint, "reviewer_r25_landlocked_joint_switch_audit.json")

    metric, s8_sensitivity, baseline = peer_tests(df, scenarios)
    save_csv(metric, "reviewer_r26_euclidean_cosine_peer_stability.csv")
    save_csv(s8_sensitivity, "reviewer_r26_s8_corrected_topn_sensitivity.csv")
    save_json(baseline, "reviewer_r26_baseline_overlap_and_egypt.json")

    summary = {
        "specification": {"n_countries": len(df), "representation": "S8-corrected eight-domain RobustScaler PCA90", "pca_components": n90, "bootstrap_iterations": N_BOOT},
        "r24": {"gap_max_k": int(gap.loc[gap.gap.idxmax(), "k"]), "gap_1se_k": int(gap.loc[gap.tibshirani_1se_selected, "k"].iloc[0]), "dip": dip.to_dict("records"), "gmm_bic_k": int(gmm.loc[gmm.bic.idxmin(), "k"]), "gmm_icl_k": int(gmm.loc[gmm.icl_bic_plus_2entropy.idxmin(), "k"]), "method_bootstrap": method_summary.to_dict("records"), "consensus_dependence": dependence.to_dict("records"), "silhouette_test": sil_summary.iloc[0].to_dict(), "method_splits": split.to_dict("records")},
        "r25": {**dist_summary, "landlocked_joint_switch_audit": land_joint},
        "r26": {"china": metric[metric.country.eq("China")].to_dict("records"), "spain": metric[metric.country.eq("Spain")].to_dict("records"), "pakistan_s8": s8_sensitivity[s8_sensitivity.country.eq("Pakistan")].to_dict("records"), "poland_s8": s8_sensitivity[s8_sensitivity.country.eq("Poland")].to_dict("records"), "baseline": baseline},
    }
    save_json(summary, "reviewer_tests_r24_r25_r26_summary.json")
    print(json.dumps(cap.native(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
