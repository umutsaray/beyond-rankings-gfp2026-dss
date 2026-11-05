from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from extract_gfp_pdfs import (
    AIRPOWER_FIELDS,
    DEFAULT_OUTPUT_DIR,
    FINANCIAL_FIELDS,
    GEOGRAPHY_FIELDS,
    LAND_FIELDS,
    LOGISTICS_FIELDS,
    MANPOWER_FIELDS,
    NATURAL_RESOURCE_FIELDS,
    NAVAL_FIELDS,
    RANK_FIELDS,
    VALUE_FIELDS,
)


RANDOM_STATE = 42
K_VALUES = range(2, 11)
PRIMARY_MISSING_LIMIT = 0.40
HIGH_CORRELATION = 0.90
OUTPUT_SUBDIR = "capability_profiles"
META_EXCLUSIONS = {
    "country", "year", "source_file", "source_url", "rank", "total_countries",
    "pwrindx", "last_reviewed_date", "extraction_engine", "text_line_count",
}
DOMAIN_FIELDS = {
    "financial": FINANCIAL_FIELDS,
    "geography": GEOGRAPHY_FIELDS,
    "manpower": MANPOWER_FIELDS,
    "airpower": AIRPOWER_FIELDS,
    "land": LAND_FIELDS,
    "naval": NAVAL_FIELDS,
    "natural_resources": NATURAL_RESOURCE_FIELDS,
    "logistics": LOGISTICS_FIELDS,
}
FIELD_DOMAIN = {field: domain for domain, fields in DOMAIN_FIELDS.items() for field in fields}
DEFICIT_FIELDS = {"oil_deficit", "natural_gas_deficit", "coal_deficit"}
BOUNDED_FIELDS = {"internet_coverage", "naval_presence"}
SCENARIOS = {
    "S1": "original_standard",
    "S2": "original_robust",
    "S3": "log_standard",
    "S4": "log_robust",
    "S5": "log_robust_winsorized",
    "S6": "log_robust_no_naval",
    "S7": "log_robust_structural_naval",
    "S8": "domain_level",
    "S9": "log_robust_with_tonnage_sensitivity",
}


@dataclass
class ScenarioData:
    scenario: str
    description: str
    features: list[str]
    frame: pd.DataFrame
    scaled: np.ndarray
    scaler: Any
    pca: PCA
    pca_scores: np.ndarray
    n_pc80: int
    n_pc90: int


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [native(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(data), indent=2, ensure_ascii=False), encoding="utf-8")


def find_turkiye(df: pd.DataFrame) -> int:
    names = df["country"].astype(str).str.casefold()
    mask = names.isin({"turkiye", "türkiye", "turkey"})
    if not mask.any() and "source_file" in df:
        mask = df["source_file"].astype(str).str.contains("Turkiye|Türkiye|Turkey", case=False, na=False)
    if not mask.any():
        raise RuntimeError("Türkiye/Turkiye/Turkey could not be located.")
    return int(np.flatnonzero(mask.to_numpy())[0])


def load_data(output_dir: Path) -> pd.DataFrame:
    clean = output_dir / "tables" / "gfp_2026_clean.csv"
    if not clean.exists():
        raise SystemExit(f"Missing {clean}; run extraction and cleaning first.")
    df = pd.read_csv(clean)
    if "country" not in df or df.empty:
        raise RuntimeError("Clean dataset has no usable country rows.")
    return df


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def build_dataset_audits(df: pd.DataFrame, tables: Path) -> dict[str, Any]:
    numeric = numeric_frame(df)
    rows = []
    for col in df.columns:
        s = numeric[col]
        observed = s.dropna()
        rows.append({
            "feature": col,
            "datatype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
            "zero_count": int((s == 0).sum()),
            "zero_pct_observed": float((s == 0).mean() * 100),
            "unique_count": int(df[col].nunique(dropna=True)),
            "min": observed.min() if len(observed) else np.nan,
            "max": observed.max() if len(observed) else np.nan,
            "mean": observed.mean() if len(observed) else np.nan,
            "median": observed.median() if len(observed) else np.nan,
            "std": observed.std() if len(observed) else np.nan,
            "skewness": observed.skew() if len(observed) > 2 else np.nan,
        })
    audit = pd.DataFrame(rows)
    save_csv(audit, tables / "dataset_audit.csv")
    save_csv(audit[["feature", "missing_count", "missing_pct", "zero_count", "zero_pct_observed"]], tables / "missingness_summary.csv")
    save_csv(audit[["feature", "unique_count", "min", "max", "mean", "median", "std", "skewness"]], tables / "descriptive_statistics.csv")
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "duplicate_countries": int(df["country"].duplicated().sum()),
        "country_names": df["country"].astype(str).tolist(),
        "total_missing": int(df.isna().sum().sum()),
    }


def naval_audit(df: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.Series]:
    naval_cols = [c for c in NAVAL_FIELDS if c in df]
    n = numeric_frame(df[naval_cols])
    coastline = pd.to_numeric(df.get("coastline"), errors="coerce")
    any_positive = n.fillna(0).gt(0).any(axis=1)
    all_absent = n.fillna(0).eq(0).all(axis=1)
    structural = coastline.eq(0) & all_absent
    classification = np.where(any_positive, "observed_naval_presence", np.where(structural, "structural_absence_landlocked", "unknown_or_unreported"))
    presence = pd.Series(np.where(any_positive, 1.0, np.where(structural, 0.0, np.nan)), index=df.index, name="naval_presence")
    out = pd.DataFrame({
        "country": df["country"], "coastline": coastline,
        "naval_observed_fields": n.notna().sum(axis=1), "naval_missing_fields": n.isna().sum(axis=1),
        "naval_positive_fields": n.gt(0).sum(axis=1), "classification": classification,
        "naval_presence": presence,
        "uncertainty_note": np.where(classification == "unknown_or_unreported", "Available GFP fields do not justify converting missing values to zero.", ""),
    })
    save_csv(out, tables / "naval_missingness_audit.csv")
    return out, presence


def choose_primary_features(df: pd.DataFrame, presence: pd.Series, tables: Path) -> tuple[list[str], pd.DataFrame]:
    decisions = []
    selected = []
    rank_set = set(RANK_FIELDS)
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        reason = ""
        include = True
        if col in META_EXCLUSIONS:
            include, reason = False, "metadata_or_posthoc_only"
        elif col in rank_set or col.endswith("_rank"):
            include, reason = False, "rank_derived"
        elif col not in VALUE_FIELDS:
            include, reason = False, "not_in_extractor_value_schema"
        elif s.notna().sum() == 0:
            include, reason = False, "non_numeric_or_all_missing"
        elif col == "naval_total_tonnage":
            include, reason = False, "excluded_primary_high_missingness_sensitivity_only"
        elif s.isna().mean() > PRIMARY_MISSING_LIMIT:
            include, reason = False, "missingness_above_40pct"
        elif s.nunique(dropna=True) <= 1:
            include, reason = False, "constant"
        elif s.var(skipna=True) <= 1e-12:
            include, reason = False, "near_zero_variance"
        if include:
            selected.append(col)
        decisions.append({
            "feature": col, "domain": FIELD_DOMAIN.get(col, "metadata"), "datatype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()), "missing_pct": float(df[col].isna().mean() * 100),
            "unique_count": int(df[col].nunique(dropna=True)), "included_primary": include,
            "exclusion_reason": reason, "transformation": "evaluated_by_scenario" if include else "none",
            "scenario_usage": "S1-S5,S7,S8; S6 excludes naval" if include else ("sensitivity_only" if col == "naval_total_tonnage" else "none"),
        })
    decisions.append({
        "feature": "naval_presence", "domain": "naval", "datatype": "derived_binary_with_unknown",
        "missing_count": int(presence.isna().sum()), "missing_pct": float(presence.isna().mean() * 100),
        "unique_count": int(presence.nunique(dropna=True)), "included_primary": False,
        "exclusion_reason": "derived_for_structural_naval_scenario", "transformation": "none", "scenario_usage": "S7,S8",
    })
    out = pd.DataFrame(decisions)
    save_csv(out, tables / "feature_decisions.csv")
    return selected, out


def impute_structurally(df: pd.DataFrame, features: list[str], naval_class: pd.Series) -> pd.DataFrame:
    x = numeric_frame(df[features]).copy()
    structural = naval_class.eq("structural_absence_landlocked")
    for col in features:
        if col in NAVAL_FIELDS:
            x.loc[structural & x[col].isna(), col] = 0.0
        median = x[col].median(skipna=True)
        x[col] = x[col].fillna(0.0 if pd.isna(median) else median)
    return x


def is_log_candidate(feature: str, s: pd.Series) -> tuple[bool, str]:
    observed = s.dropna()
    if feature in BOUNDED_FIELDS:
        return False, "binary_or_bounded"
    if feature in DEFICIT_FIELDS or (len(observed) and observed.min() < 0):
        return False, "meaningful_negative_values"
    if observed.nunique() <= 2:
        return False, "binary_or_near_binary"
    skew = observed.skew()
    if np.isfinite(skew) and skew > 1.0:
        return True, "nonnegative_right_skew_above_1"
    return False, "skew_not_material"


def transformation_table(base: pd.DataFrame, tables: Path) -> tuple[set[str], pd.DataFrame]:
    rows, log_features = [], set()
    for col in base.columns:
        s = base[col]
        flag, reason = is_log_candidate(col, s)
        transformed = np.log1p(s) if flag else s
        if flag:
            log_features.add(col)
        rows.append({
            "feature": col, "original_skew": s.skew(), "transformation": "log1p" if flag else "none",
            "transformed_skew": transformed.skew(), "reason": reason,
        })
    out = pd.DataFrame(rows)
    save_csv(out, tables / "transformation_log.csv")
    return log_features, out


def domain_scores(x: pd.DataFrame, presence: pd.Series | None = None) -> pd.DataFrame:
    z = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)
    scores = {}
    for domain, fields in DOMAIN_FIELDS.items():
        cols = [c for c in fields if c in z]
        if cols:
            scores[domain] = z[cols].mean(axis=1)
    if presence is not None:
        scores["naval_presence"] = presence.fillna(presence.median())
    return pd.DataFrame(scores, index=x.index)


def prepare_scenarios(df: pd.DataFrame, features: list[str], naval_info: pd.DataFrame, presence: pd.Series, log_features: set[str]) -> dict[str, ScenarioData]:
    base = impute_structurally(df, features, naval_info["classification"])
    scenarios = {}
    for sid, description in SCENARIOS.items():
        cols = features.copy()
        frame = base.copy()
        if sid == "S9" and "naval_total_tonnage" in df:
            tonnage = pd.to_numeric(df["naval_total_tonnage"], errors="coerce")
            structural = naval_info["classification"].eq("structural_absence_landlocked")
            tonnage.loc[structural & tonnage.isna()] = 0.0
            tonnage = tonnage.fillna(tonnage.median(skipna=True))
            frame["naval_total_tonnage"] = tonnage
            cols = list(frame.columns)
        if sid == "S6":
            cols = [c for c in cols if c not in NAVAL_FIELDS]
            frame = frame[cols]
        elif sid == "S7":
            naval_cols = [c for c in NAVAL_FIELDS if c in frame]
            intensity = np.log1p(frame[naval_cols].clip(lower=0)).mean(axis=1) if naval_cols else pd.Series(0, index=frame.index)
            frame["naval_presence"] = presence.fillna(presence.median())
            frame["naval_intensity"] = intensity
            cols = list(frame.columns)
        if sid in {"S3", "S4", "S5", "S6", "S7", "S8", "S9"}:
            for col in [c for c in frame.columns if c in log_features or c == "naval_total_tonnage"]:
                frame[col] = np.log1p(frame[col].clip(lower=0))
        if sid == "S5":
            for col in frame.columns:
                lo, hi = frame[col].quantile([0.01, 0.99])
                frame[col] = frame[col].clip(lo, hi)
        if sid == "S8":
            frame = domain_scores(frame, presence)
            cols = list(frame.columns)
        scaler = StandardScaler() if sid in {"S1", "S3"} else RobustScaler(quantile_range=(25, 75))
        scaled = scaler.fit_transform(frame)
        pca = PCA(random_state=RANDOM_STATE).fit(scaled)
        scores = pca.transform(scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n80 = int(np.searchsorted(cumulative, 0.80) + 1)
        n90 = int(np.searchsorted(cumulative, 0.90) + 1)
        scenarios[sid] = ScenarioData(sid, description, cols, frame, scaled, scaler, pca, scores, n80, n90)
    return scenarios


def correlation_audit(base: pd.DataFrame, tables: Path) -> pd.DataFrame:
    pearson = base.corr(method="pearson")
    spearman = base.corr(method="spearman")
    rows = []
    cols = list(base.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p, s = pearson.loc[a, b], spearman.loc[a, b]
            if max(abs(p), abs(s)) >= HIGH_CORRELATION:
                rows.append({"feature_1": a, "feature_2": b, "pearson": p, "spearman": s, "flag_reason": "abs_correlation_gte_0.90"})
    out = pd.DataFrame(rows).sort_values("spearman", key=lambda x: x.abs(), ascending=False)
    save_csv(out, tables / "high_correlation_pairs.csv")
    return out


def kmedoids(x: np.ndarray, k: int, seed: int = RANDOM_STATE, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    dist = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2), 0))
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        nearest = dist[:, medoids].min(axis=1)
        probs = nearest ** 2
        probs[medoids] = 0
        medoids.append(int(rng.choice(n, p=probs / probs.sum())))
    medoids = np.array(medoids, dtype=int)
    for _ in range(max_iter):
        labels = dist[:, medoids].argmin(axis=1)
        new = medoids.copy()
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx):
                new[c] = idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))]
        if np.array_equal(new, medoids):
            break
        medoids = new
    labels = dist[:, medoids].argmin(axis=1)
    return labels, medoids, float(dist[np.arange(n), medoids[labels]].sum())


def fit_labels(x: np.ndarray, algorithm: str, k: int, seed: int = RANDOM_STATE) -> tuple[np.ndarray, dict[str, float]]:
    extra: dict[str, float] = {}
    if algorithm == "kmeans":
        model = KMeans(k, n_init=30, random_state=seed).fit(x)
        extra["inertia"] = float(model.inertia_)
        return model.labels_, extra
    if algorithm.startswith("agglomerative_"):
        linkage = algorithm.split("_", 1)[1]
        return AgglomerativeClustering(k, linkage=linkage).fit_predict(x), extra
    if algorithm == "gmm":
        model = GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-5, random_state=seed).fit(x)
        extra.update({"aic": float(model.aic(x)), "bic": float(model.bic(x))})
        return model.predict(x), extra
    if algorithm == "kmedoids":
        labels, _, cost = kmedoids(x, k, seed)
        extra["medoid_cost"] = cost
        return labels, extra
    raise ValueError(algorithm)


def representation_arrays(s: ScenarioData) -> dict[str, np.ndarray]:
    if s.scenario == "S8":
        return {"domain": s.scaled, "pca90": s.pca_scores[:, :s.n_pc90]}
    return {"original": s.scaled, "pca80": s.pca_scores[:, :s.n_pc80], "pca90": s.pca_scores[:, :s.n_pc90]}


def validate_all(scenarios: dict[str, ScenarioData], tables: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, str, int], np.ndarray]]:
    algorithms = ["kmeans", "agglomerative_ward", "agglomerative_average", "agglomerative_complete", "gmm", "kmedoids"]
    rows, labels_cache = [], {}
    for sid, scenario in scenarios.items():
        for rep, x in representation_arrays(scenario).items():
            for algorithm in algorithms:
                for k in K_VALUES:
                    try:
                        labels, extra = fit_labels(x, algorithm, k)
                        sizes = np.bincount(labels, minlength=k)
                        if len(np.unique(labels)) < 2:
                            raise ValueError("single cluster returned")
                        row = {
                            "scenario": sid, "scenario_description": scenario.description, "representation": rep,
                            "algorithm": algorithm, "k": k, "silhouette": silhouette_score(x, labels),
                            "calinski_harabasz": calinski_harabasz_score(x, labels),
                            "davies_bouldin": davies_bouldin_score(x, labels), "cluster_sizes": "|".join(map(str, sorted(sizes))),
                            "smallest_cluster_size": int(sizes.min()), "largest_cluster_size": int(sizes.max()),
                            "smallest_cluster_proportion": float(sizes.min() / len(labels)),
                            "imbalance_ratio": float(sizes.max() / max(sizes.min(), 1)), **extra,
                        }
                        rows.append(row)
                        labels_cache[(sid, rep, algorithm, k)] = labels
                    except Exception as exc:
                        rows.append({"scenario": sid, "scenario_description": scenario.description, "representation": rep, "algorithm": algorithm, "k": k, "error": str(exc)})
    out = pd.DataFrame(rows)
    save_csv(out, tables / "cluster_validation_all.csv")
    return out, labels_cache


def preliminary_stability(x: np.ndarray, algorithm: str, k: int, base: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    aris, amis, nmis = [], [], []
    n = len(x)
    for b in range(resamples):
        idx = rng.choice(n, n, replace=True)
        unique = np.unique(idx)
        if len(unique) <= k:
            continue
        try:
            sub, _ = fit_labels(x[unique], algorithm, k, seed + b + 1)
            aris.append(adjusted_rand_score(base[unique], sub))
            amis.append(adjusted_mutual_info_score(base[unique], sub))
            nmis.append(normalized_mutual_info_score(base[unique], sub))
        except Exception:
            continue
    return float(np.mean(aris)), float(np.mean(amis)), float(np.mean(nmis))


def candidate_summary(validation: pd.DataFrame, scenarios: dict[str, ScenarioData], cache: dict, mode: str, tables: Path) -> tuple[pd.DataFrame, tuple[str, str, str, int]]:
    valid = validation.dropna(subset=["silhouette"]).copy()
    valid["balance_flag"] = np.where((valid.smallest_cluster_size < 5) | (valid.smallest_cluster_proportion < 0.035) | (valid.imbalance_ratio > 12), "trivial_or_outlier_dominated", "eligible")
    shortlist = pd.concat([
        valid[valid.balance_flag.eq("eligible")].nlargest(15, "silhouette"),
        valid[valid.balance_flag.eq("eligible")].nsmallest(10, "davies_bouldin"),
        valid[valid.balance_flag.eq("eligible")].nlargest(10, "calinski_harabasz"),
    ]).drop_duplicates(["scenario", "representation", "algorithm", "k"])
    stability_rows = []
    prelim_n = 8 if mode == "fast" else 20
    for _, r in shortlist.iterrows():
        key = (r.scenario, r.representation, r.algorithm, int(r.k))
        x = representation_arrays(scenarios[r.scenario])[r.representation]
        ari, ami, nmi = preliminary_stability(x, r.algorithm, int(r.k), cache[key], prelim_n, RANDOM_STATE + len(stability_rows) * 17)
        stability_rows.append({"scenario": r.scenario, "representation": r.representation, "algorithm": r.algorithm, "k": int(r.k), "bootstrap_ARI_preliminary": ari, "bootstrap_AMI_preliminary": ami, "bootstrap_NMI_preliminary": nmi})
    stab = pd.DataFrame(stability_rows)
    out = valid.merge(stab, on=["scenario", "representation", "algorithm", "k"], how="left")
    method_agreement = {}
    for (sid, rep, k), group in out.groupby(["scenario", "representation", "k"]):
        label_sets = [cache[(sid, rep, a, int(k))] for a in group.algorithm if (sid, rep, a, int(k)) in cache]
        pairs = [adjusted_rand_score(label_sets[i], label_sets[j]) for i in range(len(label_sets)) for j in range(i + 1, len(label_sets))]
        method_agreement[(sid, rep, int(k))] = float(np.mean(pairs)) if pairs else np.nan
    out["cross_method_ARI"] = [method_agreement.get((r.scenario, r.representation, int(r.k)), np.nan) for _, r in out.iterrows()]
    out["candidate_flag"] = out.balance_flag
    plausible = out[(out.balance_flag == "eligible") & out.bootstrap_ARI_preliminary.notna()].copy()
    if plausible.empty:
        plausible = out[out.balance_flag == "eligible"].copy()
    # Transparent lexicographic rule: realism filter, then stability, cross-method agreement,
    # silhouette, and Davies-Bouldin. No hidden weighted composite score.
    plausible = plausible.sort_values(["bootstrap_ARI_preliminary", "cross_method_ARI", "silhouette", "davies_bouldin"], ascending=[False, False, False, True])
    preferred_row = plausible.iloc[0]
    preferred = (preferred_row.scenario, preferred_row.representation, preferred_row.algorithm, int(preferred_row.k))
    out.loc[(out.scenario == preferred[0]) & (out.representation == preferred[1]) & (out.algorithm == preferred[2]) & (out.k == preferred[3]), "candidate_flag"] = "plausible_preferred"
    out.loc[(out.balance_flag == "eligible") & (out.candidate_flag != "plausible_preferred"), "candidate_flag"] = "questionable_or_secondary"
    save_csv(out, tables / "cluster_candidate_summary.csv")
    return out, preferred


def full_bootstrap(x: np.ndarray, algorithm: str, k: int, base: np.ndarray, countries: list[str], resamples: int, tables: Path, variant_labels: list[np.ndarray] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE + 900)
    n = len(x)
    rows, coassoc = [], np.zeros((n, n), float)
    denom = np.zeros((n, n), float)
    observation_hits = np.zeros(n)
    observation_total = np.zeros(n)
    for b in range(resamples):
        idx = rng.choice(n, n, replace=True)
        unique = np.unique(idx)
        if len(unique) <= k:
            continue
        labels, _ = fit_labels(x[unique], algorithm, k, RANDOM_STATE + 1000 + b)
        ari = adjusted_rand_score(base[unique], labels)
        ami = adjusted_mutual_info_score(base[unique], labels)
        nmi = normalized_mutual_info_score(base[unique], labels)
        cluster_jaccards = []
        for baseline_cluster in np.unique(base[unique]):
            baseline_set = set(np.where(base[unique] == baseline_cluster)[0])
            best = 0.0
            for sampled_cluster in np.unique(labels):
                sampled_set = set(np.where(labels == sampled_cluster)[0])
                union = baseline_set | sampled_set
                best = max(best, len(baseline_set & sampled_set) / len(union) if union else 1.0)
            cluster_jaccards.append(best)
        rows.append({"resample": b + 1, "unique_countries": len(unique), "ARI": ari, "AMI": ami, "NMI": nmi, "cluster_jaccard_mean": np.mean(cluster_jaccards), "cluster_jaccard_min": np.min(cluster_jaccards)})
        same = labels[:, None] == labels[None, :]
        coassoc[np.ix_(unique, unique)] += same
        denom[np.ix_(unique, unique)] += 1
        # Membership recurrence is pairwise recurrence within each baseline cluster.
        for local, original in enumerate(unique):
            peers = np.where(base[unique] == base[original])[0]
            observation_hits[original] += np.mean(labels[peers] == labels[local]) if len(peers) else 1
            observation_total[original] += 1
    for variant in variant_labels or []:
        coassoc += variant[:, None] == variant[None, :]
        denom += 1
    consensus = np.divide(coassoc, denom, out=np.zeros_like(coassoc), where=denom > 0)
    obs = np.divide(observation_hits, observation_total, out=np.zeros_like(observation_hits), where=observation_total > 0)
    summary = pd.DataFrame(rows)
    summary["mean_ARI"] = summary.ARI.mean() if len(summary) else np.nan
    summary["mean_AMI"] = summary.AMI.mean() if len(summary) else np.nan
    summary["mean_NMI"] = summary.NMI.mean() if len(summary) else np.nan
    save_csv(summary, tables / "bootstrap_stability.csv")
    save_csv(pd.DataFrame(consensus, index=countries, columns=countries).reset_index(names="country"), tables / "consensus_matrix.csv")
    save_csv(pd.DataFrame({"country": countries, "observation_membership_stability": obs}), tables / "observation_stability.csv")
    return summary, consensus, obs


def pca_outputs(scenario: ScenarioData, countries: pd.Series, tables: Path) -> None:
    ev = pd.DataFrame({
        "component": np.arange(1, len(scenario.pca.explained_variance_) + 1),
        "eigenvalue": scenario.pca.explained_variance_,
        "explained_variance_ratio": scenario.pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(scenario.pca.explained_variance_ratio_),
        "retained_80pct": np.arange(1, len(scenario.pca.explained_variance_) + 1) <= scenario.n_pc80,
        "retained_90pct": np.arange(1, len(scenario.pca.explained_variance_) + 1) <= scenario.n_pc90,
    })
    loadings = pd.DataFrame(scenario.pca.components_.T, index=scenario.features, columns=[f"PC{i+1}" for i in range(scenario.pca.n_components_)]).reset_index(names="feature")
    scores = pd.DataFrame(scenario.pca_scores, columns=[f"PC{i+1}" for i in range(scenario.pca.n_components_)])
    scores.insert(0, "country", countries.to_numpy())
    save_csv(ev, tables / "pca_explained_variance.csv")
    save_csv(loadings, tables / "pca_loadings.csv")
    save_csv(scores, tables / "pca_scores.csv")


def outlier_analysis(x: np.ndarray, countries: pd.Series, preferred: tuple, base_labels: np.ndarray, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=min(10, x.shape[1]), random_state=RANDOM_STATE).fit_transform(x)
    center = np.median(pca, axis=0)
    cov = LedoitWolf().fit(pca)
    robust_distance = np.sqrt(np.maximum(cov.mahalanobis(pca), 0))
    pca_distance = np.linalg.norm(pca - center, axis=1)
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE).fit(x)
    out = pd.DataFrame({"country": countries, "pca_score_distance": pca_distance, "shrinkage_mahalanobis": robust_distance, "isolation_forest_score": -iso.score_samples(x), "isolation_forest_flag": iso.predict(x) == -1})
    out["extreme_rank"] = out[["pca_score_distance", "shrinkage_mahalanobis", "isolation_forest_score"]].rank(pct=True).mean(axis=1)
    out = out.sort_values("extreme_rank", ascending=False)
    save_csv(out, tables / "outlier_diagnostics.csv")
    sid, rep, algorithm, k = preferred
    rows = []
    extreme_idx = out.head(5).index.to_numpy()
    tests = [(f"leave_{countries.iloc[i]}_out", np.array([i])) for i in extreme_idx] + [("leave_extreme_set_out", extreme_idx)]
    for name, removed in tests:
        keep = np.setdiff1d(np.arange(len(x)), removed)
        labels, _ = fit_labels(x[keep], algorithm, k, RANDOM_STATE + 77)
        rows.append({"analysis": name, "removed_countries": "|".join(countries.iloc[removed]), "remaining_n": len(keep), "ARI": adjusted_rand_score(base_labels[keep], labels), "AMI": adjusted_mutual_info_score(base_labels[keep], labels)})
    sens = pd.DataFrame(rows)
    save_csv(sens, tables / "outlier_sensitivity.csv")
    return out, sens


def pairwise_distances(x: np.ndarray, tr: int, metric: str) -> np.ndarray:
    delta = x - x[tr]
    if metric == "euclidean":
        return np.linalg.norm(delta, axis=1)
    if metric == "cosine":
        denom = np.linalg.norm(x, axis=1) * max(np.linalg.norm(x[tr]), 1e-12)
        return 1 - np.divide(x @ x[tr], denom, out=np.zeros(len(x)), where=denom > 0)
    if metric == "mahalanobis":
        precision = LedoitWolf().fit(x).precision_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0))
    raise ValueError(metric)


def peer_analysis(df: pd.DataFrame, scenarios: dict[str, ScenarioData], tr: int, candidate_labels: list[np.ndarray], tables: Path, bootstrap_runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sid, s in scenarios.items():
        spaces = {"original": s.scaled, "pca90": s.pca_scores[:, :s.n_pc90]}
        for space, x in spaces.items():
            for metric in ["euclidean", "mahalanobis", "cosine"]:
                d = pairwise_distances(x, tr, metric)
                order = np.argsort(d)
                order = order[order != tr]
                max_d = max(float(np.nanmax(d[order])), 1e-12)
                for rank_pos, idx in enumerate(order[:20], 1):
                    rows.append({
                        "scenario": sid, "space": space, "metric": metric, "country": df.country.iloc[idx],
                        "neighbor_rank": rank_pos, "distance": d[idx], "normalized_distance": d[idx] / max_d,
                        "in_top5": rank_pos <= 5, "in_top10": rank_pos <= 10, "in_top20": True,
                    })
    rng = np.random.default_rng(RANDOM_STATE + 4444)
    bootstrap_base = scenarios["S4"].scaled
    for b in range(bootstrap_runs):
        column_idx = rng.choice(bootstrap_base.shape[1], bootstrap_base.shape[1], replace=True)
        x = bootstrap_base[:, column_idx]
        d = pairwise_distances(x, tr, "euclidean")
        order = np.argsort(d)
        order = order[order != tr]
        max_d = max(float(np.nanmax(d[order])), 1e-12)
        for rank_pos, idx in enumerate(order[:20], 1):
            rows.append({"scenario": f"feature_bootstrap_{b+1}", "space": "original", "metric": "euclidean", "country": df.country.iloc[idx], "neighbor_rank": rank_pos, "distance": d[idx], "normalized_distance": d[idx] / max_d, "in_top5": rank_pos <= 5, "in_top10": rank_pos <= 10, "in_top20": True})
    all_neighbors = pd.DataFrame(rows)
    save_csv(all_neighbors, tables / "turkiye_neighbors_all.csv")
    total_runs = all_neighbors[["scenario", "space", "metric"]].drop_duplicates().shape[0]
    same_counts = defaultdict(list)
    for labels in candidate_labels:
        for idx in range(len(df)):
            if idx != tr:
                same_counts[df.country.iloc[idx]].append(float(labels[idx] == labels[tr]))
    grouped = all_neighbors.groupby("country")
    stable = grouped.agg(appearance_count=("in_top10", "sum"), mean_neighbor_rank=("neighbor_rank", "mean"), median_neighbor_rank=("neighbor_rank", "median"), mean_normalized_distance=("normalized_distance", "mean")).reset_index()
    stable["eligible_runs"] = total_runs
    stable["appearance_frequency"] = stable.appearance_count / total_runs
    stable["same_cluster_frequency"] = stable.country.map(lambda c: np.mean(same_counts[c]) if same_counts[c] else np.nan)
    stable = stable.sort_values(["appearance_frequency", "mean_neighbor_rank"], ascending=[False, True])
    save_csv(stable, tables / "turkiye_peer_stability.csv")
    return all_neighbors, stable


def cluster_profiles_and_turkiye(df: pd.DataFrame, scenario: ScenarioData, labels: np.ndarray, tr: int, obs_stability: np.ndarray, umap_coords: np.ndarray, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = pd.DataFrame(StandardScaler().fit_transform(scenario.frame), columns=scenario.features, index=df.index)
    domain = scenario.frame.copy() if scenario.scenario == "S8" else domain_scores(scenario.frame)
    domain_z = pd.DataFrame(StandardScaler().fit_transform(domain), columns=domain.columns, index=df.index)
    rows = []
    for cluster in sorted(np.unique(labels)):
        idx = labels == cluster
        row = {"cluster": f"Cluster {cluster + 1}", "cluster_id": int(cluster), "size": int(idx.sum())}
        row.update({f"mean_z_{c}": z.loc[idx, c].mean() for c in z.columns})
        row.update({f"median_{c}": scenario.frame.loc[idx, c].median() for c in scenario.features})
        row.update({f"domain_z_{c}": domain_z.loc[idx, c].mean() for c in domain_z.columns})
        rows.append(row)
    profiles = pd.DataFrame(rows)
    save_csv(profiles, tables / "cluster_profiles.csv")
    dom = domain_z.copy()
    dom.insert(0, "country", df.country)
    dom["cluster"] = [f"Cluster {x + 1}" for x in labels]
    save_csv(dom, tables / "domain_profiles.csv")
    cluster = labels[tr]
    centroid = z.loc[labels == cluster].mean(axis=0)
    deviation = z.loc[tr] - centroid
    tr_row = {
        "country": df.country.iloc[tr], "cluster": f"Cluster {cluster + 1}",
        "membership_stability": obs_stability[tr], "pca1": scenario.pca_scores[tr, 0],
        "pca2": scenario.pca_scores[tr, 1] if scenario.pca_scores.shape[1] > 1 else np.nan,
        "umap1": umap_coords[tr, 0], "umap2": umap_coords[tr, 1],
        "distance_to_centroid": np.linalg.norm(deviation),
        "largest_positive_deviations": "|".join(deviation.nlargest(5).index),
        "largest_negative_deviations": "|".join(deviation.nsmallest(5).index),
    }
    tr_row.update({f"z_{c}": z.loc[tr, c] for c in z.columns})
    tr_row.update({f"domain_z_{c}": domain_z.loc[tr, c] for c in domain_z.columns})
    turkiye = pd.DataFrame([tr_row])
    save_csv(turkiye, tables / "turkiye_profile.csv")
    return profiles, dom, turkiye


def rank_profile_comparison(df: pd.DataFrame, tr: int, x: np.ndarray, peers: pd.DataFrame, tables: Path) -> pd.DataFrame:
    dist = pairwise_distances(x, tr, "euclidean")
    rank = pd.to_numeric(df["rank"], errors="coerce")
    out = pd.DataFrame({"country": df.country, "gfp_rank": rank, "gfp_rank_difference": (rank - rank.iloc[tr]).abs(), "profile_distance": dist})
    freq = peers.set_index("country")["appearance_frequency"]
    out["peer_recurrence_frequency"] = out.country.map(freq).fillna(0)
    mask = out.index != tr
    association = out.loc[mask, ["gfp_rank_difference", "profile_distance"]].corr(method="spearman").iloc[0, 1]
    out["spearman_rank_difference_vs_profile_distance"] = association
    out["interpretation_warning"] = "profile similarity != equal military power; geopolitical alignment; alliance; or combat equivalence"
    save_csv(out, tables / "rank_vs_profile.csv")
    return out


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1)
    return out


def statistical_tests(frame: pd.DataFrame, labels: np.ndarray, tables: Path) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        groups = [frame.loc[labels == c, col].dropna().to_numpy() for c in np.unique(labels)]
        if any(len(g) < 2 for g in groups):
            continue
        try:
            f, p_anova = stats.f_oneway(*groups)
            h, p_kw = stats.kruskal(*groups)
            overall = np.concatenate(groups)
            between = sum(len(g) * (g.mean() - overall.mean()) ** 2 for g in groups)
            total_ss = sum(float(((g - overall.mean()) ** 2).sum()) for g in groups)
            eta2 = between / max(total_ss, 1e-12)
            rows.append({"feature": col, "test": "ANOVA", "statistic": f, "p_value": p_anova, "effect_size": eta2, "effect_size_name": "eta_squared"})
            welch_f, welch_p = stats.f_oneway(*groups, equal_var=False)
            rows.append({"feature": col, "test": "Welch ANOVA", "statistic": welch_f, "p_value": welch_p, "effect_size": eta2, "effect_size_name": "eta_squared_descriptive"})
            rows.append({"feature": col, "test": "Kruskal-Wallis", "statistic": h, "p_value": p_kw, "effect_size": (h - len(groups) + 1) / max(len(overall) - len(groups), 1), "effect_size_name": "epsilon_squared"})
            if p_kw < 0.05:
                unique_labels = sorted(np.unique(labels))
                for i, left in enumerate(unique_labels):
                    for right in unique_labels[i + 1:]:
                        a, b = frame.loc[labels == left, col], frame.loc[labels == right, col]
                        u, post_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                        rank_biserial = 1 - 2 * u / (len(a) * len(b))
                        rows.append({"feature": col, "test": f"Mann-Whitney posthoc Cluster {left+1} vs {right+1}", "statistic": u, "p_value": post_p, "effect_size": rank_biserial, "effect_size_name": "rank_biserial"})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_bh"] = bh_adjust(out.p_value.fillna(1))
    save_csv(out, tables / "cluster_statistical_tests.csv")
    return out


def domain_deletion(scenario: ScenarioData, labels: np.ndarray, algorithm: str, k: int, tables: Path) -> pd.DataFrame:
    rows = []
    if scenario.scenario == "S8":
        for column_index, domain in enumerate(scenario.features):
            keep = [i for i in range(len(scenario.features)) if i != column_index]
            if len(keep) < 2:
                continue
            new, _ = fit_labels(scenario.scaled[:, keep], algorithm, k, RANDOM_STATE + 303)
            rows.append({"deleted_domain": domain, "remaining_features": len(keep), "ARI": adjusted_rand_score(labels, new), "AMI": adjusted_mutual_info_score(labels, new)})
        out = pd.DataFrame(rows)
        save_csv(out, tables / "domain_deletion_stability.csv")
        return out
    for domain, fields in DOMAIN_FIELDS.items():
        keep = [i for i, c in enumerate(scenario.features) if c not in fields]
        if len(keep) < 2:
            continue
        new, _ = fit_labels(scenario.scaled[:, keep], algorithm, k, RANDOM_STATE + 303)
        rows.append({"deleted_domain": domain, "remaining_features": len(keep), "ARI": adjusted_rand_score(labels, new), "AMI": adjusted_mutual_info_score(labels, new)})
    out = pd.DataFrame(rows)
    save_csv(out, tables / "domain_deletion_stability.csv")
    return out


def supplementary_umap(x: np.ndarray, warnings_list: list[dict[str, str]]) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, metric="euclidean", random_state=RANDOM_STATE, n_jobs=1).fit_transform(x)
    except Exception as exc:
        warnings_list.append({"code": "umap_failed", "message": str(exc)})
        return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def create_figures(df: pd.DataFrame, scenario: ScenarioData, labels: np.ndarray, tr: int, validation: pd.DataFrame, corr: pd.DataFrame, consensus: np.ndarray, outliers: pd.DataFrame, domains: pd.DataFrame, peers: pd.DataFrame, rank_profile: pd.DataFrame, candidate: pd.DataFrame, umap_coords: np.ndarray, figures: Path) -> dict[str, str]:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    made = {}
    fig, ax = plt.subplots(figsize=(12, 2.5)); ax.axis("off")
    ax.text(.5,.75,"GFP PDFs → Clean data → Audits & scenarios → PCA retained space → Multi-algorithm clustering",ha="center",fontsize=11,weight="bold")
    ax.text(.5,.35,"Validation + stability + consensus + outliers → Türkiye profile & recurring peers",ha="center",fontsize=11)
    save_figure(fig, figures/"fig01_analysis_workflow"); made["workflow"]="fig01_analysis_workflow"

    fig, ax = plt.subplots(figsize=(10,8)); subset = scenario.frame.corr(method="spearman"); sns.heatmap(subset, cmap="vlag", center=0, xticklabels=False, yticklabels=False, ax=ax); ax.set_title("Feature redundancy: Spearman correlation"); save_figure(fig, figures/"fig02_correlation_redundancy"); made["correlation"]="fig02_correlation_redundancy"

    cum=np.cumsum(scenario.pca.explained_variance_ratio_); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(np.arange(1,len(cum)+1),scenario.pca.explained_variance_ratio_,alpha=.6); ax.plot(np.arange(1,len(cum)+1),cum,color="crimson"); ax.axhline(.8,ls="--"); ax.axhline(.9,ls=":"); ax.set(xlabel="Principal component",ylabel="Variance ratio",title="PCA explained and cumulative variance"); save_figure(fig,figures/"fig03_pca_scree"); made["pca_scree"]="fig03_pca_scree"

    def map_plot(coords,title,name):
        fig,ax=plt.subplots(figsize=(9,7)); sns.scatterplot(x=coords[:,0],y=coords[:,1],hue=[f"Cluster {x+1}" for x in labels],palette="tab10",s=55,ax=ax); ax.scatter(coords[tr,0],coords[tr,1],marker="*",s=240,c="red",edgecolor="white"); ax.annotate("Türkiye",coords[tr],xytext=(8,8),textcoords="offset points",weight="bold"); ax.set_title(title); save_figure(fig,figures/name); made[name]=name
    map_plot(scenario.pca_scores[:,:2],"PCA capability map (two PCs shown; clustering used retained space)","fig04_pca_capability_map")
    map_plot(umap_coords,"Supplementary UMAP map (visualization only)","fig05_umap_map")

    fig,axes=plt.subplots(1,3,figsize=(15,4)); v=validation[validation.algorithm.eq("kmeans")]; sns.lineplot(v,x="k",y="silhouette",hue="scenario",legend=False,ax=axes[0]); sns.lineplot(v,x="k",y="calinski_harabasz",hue="scenario",legend=False,ax=axes[1]); sns.lineplot(v,x="k",y="davies_bouldin",hue="scenario",ax=axes[2]); axes[0].set_title("Silhouette"); axes[1].set_title("Calinski-Harabasz"); axes[2].set_title("Davies-Bouldin"); save_figure(fig,figures/"fig06_cluster_validation"); made["validation"]="fig06_cluster_validation"

    domain_cols=[c for c in domains if c not in {"country","cluster"}]; dm=domains.groupby("cluster")[domain_cols].mean(); fig,ax=plt.subplots(figsize=(9,5)); sns.heatmap(dm,cmap="vlag",center=0,annot=True,fmt=".2f",ax=ax); ax.set_title("Standardized cluster-domain profiles"); save_figure(fig,figures/"fig07_cluster_domain_heatmap"); made["domains"]="fig07_cluster_domain_heatmap"
    order=np.argsort(labels); fig,ax=plt.subplots(figsize=(9,8)); sns.heatmap(consensus[np.ix_(order,order)],cmap="viridis",vmin=0,vmax=1,xticklabels=False,yticklabels=False,ax=ax); ax.set_title("Consensus/co-association matrix"); save_figure(fig,figures/"fig08_consensus_heatmap"); made["consensus"]="fig08_consensus_heatmap"
    fig,ax=plt.subplots(figsize=(8,6)); sns.scatterplot(outliers,x="pca_score_distance",y="shrinkage_mahalanobis",hue="isolation_forest_flag",ax=ax); [ax.annotate(r.country,(r.pca_score_distance,r.shrinkage_mahalanobis),fontsize=7) for _,r in outliers.head(8).iterrows()]; ax.set_title("Outlier diagnostics"); save_figure(fig,figures/"fig09_outlier_diagnostics"); made["outliers"]="fig09_outlier_diagnostics"

    z=pd.DataFrame(StandardScaler().fit_transform(scenario.frame),columns=scenario.features); dev=(z.iloc[tr]-z.loc[labels==labels[tr]].mean()).sort_values(); sel=pd.concat([dev.head(6),dev.tail(6)]); fig,ax=plt.subplots(figsize=(9,6)); sel.plot.barh(color=np.where(sel>0,"#2563eb","#dc2626"),ax=ax); ax.set_title("Türkiye deviations from its cluster centroid"); ax.set_xlabel("Standardized deviation"); save_figure(fig,figures/"fig10_turkiye_centroid_profile"); made["turkiye_profile"]="fig10_turkiye_centroid_profile"
    top=peers.head(15).sort_values("appearance_frequency"); fig,ax=plt.subplots(figsize=(9,6)); ax.barh(top.country,top.appearance_frequency,color="#0f766e"); ax.set(xlabel="Top-10 appearance frequency",title="Türkiye recurring structural peers"); save_figure(fig,figures/"fig11_turkiye_stable_peers"); made["peers"]="fig11_turkiye_stable_peers"
    rp=rank_profile[rank_profile.country.ne(df.country.iloc[tr])]; fig,ax=plt.subplots(figsize=(8,6)); sns.scatterplot(rp,x="gfp_rank_difference",y="profile_distance",size="peer_recurrence_frequency",sizes=(15,180),alpha=.7,ax=ax); ax.set_title("GFP rank proximity versus profile proximity"); save_figure(fig,figures/"fig12_rank_vs_profile"); made["rank_profile"]="fig12_rank_vs_profile"
    cs=candidate.dropna(subset=["silhouette"]).copy(); fig,ax=plt.subplots(figsize=(9,6)); sns.scatterplot(cs,x="silhouette",y="imbalance_ratio",hue="candidate_flag",size="bootstrap_ARI_preliminary",sizes=(15,180),ax=ax); ax.set_yscale("log"); ax.set_title("Robustness and candidate realism"); save_figure(fig,figures/"fig13_sensitivity_summary"); made["sensitivity"]="fig13_sensitivity_summary"
    return made


def report_text(dataset: dict, feature_count: int, preferred: tuple, candidate_row: pd.Series, bootstrap: pd.DataFrame, turkiye: pd.DataFrame, peers: pd.DataFrame, warnings_list: list[dict[str,str]]) -> str:
    sid, rep, algorithm, k = preferred
    top_peers = ", ".join(peers.head(10).country.astype(str))
    return f"""# Mapping Global Military Capability Profiles Using Unsupervised Machine Learning

## Dataset construction
The analysis reused the cleaned Global Firepower 2026 dataset. It contains {dataset['rows']} countries and {dataset['columns']} columns, with {dataset['duplicate_countries']} duplicate country records. Existing extraction, cleaning, validation, and prior article outputs were not overwritten.

## Feature decisions and missingness
The primary matrix contains {feature_count} schema-backed capability variables. Country identifiers, year/source metadata, GFP rank, PwrIndx, every extractor-defined rank field, constants, and near-zero-variance fields were excluded. `naval_total_tonnage` was excluded from primary clustering because of excessive missingness and retained only as a documented sensitivity field.

Naval missingness was classified as observed presence, defensible structural absence for landlocked zero-capability cases, or unknown/unreported. Unknown naval values were not blindly converted to zero. Scenario S7 adds dual encoding through naval presence and naval intensity.

## Preprocessing and dimensionality reduction
Eight scenarios compare original/log1p representations, StandardScaler/RobustScaler, winsorization, naval exclusion, structural naval encoding, and domain-level scores. Log transformation was restricted to non-negative, materially right-skewed magnitude variables. PCA was fitted separately per scenario; 80% and 90% retained-variance spaces were used for clustering, while PC1-PC2 were used only for visualization. UMAP is supplementary visualization only.

## Candidate cluster evidence
K-Means, agglomerative hierarchical clustering (Ward, average, complete), Gaussian mixture models, and K-Medoids were evaluated for k=2..10 using silhouette, Calinski-Harabasz, Davies-Bouldin, cluster sizes, imbalance, preliminary stability, and cross-method agreement.

The preferred candidate is `{sid}/{rep}/{algorithm}/k={k}`. It passed the explicit realism filter and was selected lexicographically by bootstrap stability, cross-method agreement, silhouette, and Davies-Bouldin; no hidden weighted score was used. Its silhouette is {candidate_row.silhouette:.3f}, smallest cluster size is {int(candidate_row.smallest_cluster_size)}, and imbalance ratio is {candidate_row.imbalance_ratio:.2f}. High-silhouette solutions with tiny or severely imbalanced clusters were marked `trivial_or_outlier_dominated` rather than automatically selected.

## Stability, consensus, and outliers
The final bootstrap used {len(bootstrap)} successful resamples. Mean ARI={bootstrap.ARI.mean():.3f}, AMI={bootstrap.AMI.mean():.3f}, and NMI={bootstrap.NMI.mean():.3f}. Consensus and observation-level recurrence are reported separately. Extreme states were retained in the primary analysis and tested through leave-one-extreme and leave-extreme-set-out sensitivity runs.

## Türkiye's structural position
Türkiye belongs to {turkiye.iloc[0]['cluster']} with estimated membership stability {turkiye.iloc[0]['membership_stability']:.3f}. Its reported recurring peers are: {top_peers}. These peers are based on multivariate distances across preprocessing scenarios, original/PCA spaces, and Euclidean, shrinkage-Mahalanobis, and cosine metrics—not on GFP rank or PwrIndx.

Profile similarity is not equal military power, geopolitical alignment, alliance membership, or combat equivalence.

## Rank versus profile
GFP rank and PwrIndx were used only after clustering for descriptive comparison. Rank proximity and capability-profile proximity answer different questions and should not be interpreted interchangeably.

## Statistical characterization
Cluster differences were characterized using ANOVA and Kruskal-Wallis tests with Benjamini-Hochberg correction and effect sizes. Results are descriptive for a cross-sectional, secondary-source dataset and do not establish causal or combat-effectiveness claims.

## Limitations and warnings
The study is limited to one year of GFP data, opaque GFP source construction, strong feature redundancy, possible uncertainty in naval reporting, sensitivity to extreme states, and the absence of direct indicators for doctrine, technology quality, interoperability, and battlefield effectiveness.

Warnings recorded: {len(warnings_list)}. See `json/warnings.json` for machine-readable details.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust multidimensional military capability profile analysis.")
    parser.add_argument("--mode", choices=["fast", "final"], default="fast")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir / OUTPUT_SUBDIR
    tables, figures, json_dir, reports, diagnostics = [root / x for x in ["tables", "figures", "json", "reports", "diagnostics"]]
    for p in [tables, figures, json_dir, reports, diagnostics]: p.mkdir(parents=True, exist_ok=True)
    warnings_list: list[dict[str, str]] = []
    df = load_data(args.output_dir)
    dataset = build_dataset_audits(df, tables)
    if len(df) != 143: warnings_list.append({"code":"country_count_unexpected","message":f"Expected 143, found {len(df)}"})
    if df.country.duplicated().any(): warnings_list.append({"code":"duplicate_countries","message":"Duplicate country names detected."})
    tr = find_turkiye(df)
    naval, presence = naval_audit(df, tables)
    if (naval.classification == "unknown_or_unreported").any(): warnings_list.append({"code":"structural_naval_ambiguity","message":f"{int((naval.classification == 'unknown_or_unreported').sum())} countries have uncertain naval absence/reporting."})
    features, decisions = choose_primary_features(df, presence, tables)
    base = impute_structurally(df, features, naval.classification)
    log_features, transformations = transformation_table(base, tables)
    high_corr = correlation_audit(base, tables)
    if len(high_corr): warnings_list.append({"code":"high_feature_redundancy","message":f"{len(high_corr)} feature pairs have |Pearson| or |Spearman| >= 0.90."})
    scenarios = prepare_scenarios(df, features, naval, presence, log_features)
    validation, cache = validate_all(scenarios, tables)
    candidates, preferred = candidate_summary(validation, scenarios, cache, args.mode, tables)
    sid, rep, algorithm, k = preferred
    preferred_s = scenarios[sid]
    preferred_x = representation_arrays(preferred_s)[rep]
    labels = cache[preferred]
    sizes = np.bincount(labels)
    if sizes.min() < 5: warnings_list.append({"code":"tiny_cluster","message":f"Preferred solution has minimum cluster size {sizes.min()}."})
    if sizes.max()/sizes.min() > 12: warnings_list.append({"code":"severe_imbalance","message":"Preferred solution remains severely imbalanced."})
    pca_outputs(preferred_s, df.country, tables)
    n_boot = 50 if args.mode == "fast" else 500
    plausible_keys = candidates[candidates.candidate_flag.isin(["plausible_preferred","questionable_or_secondary"])].sort_values("bootstrap_ARI_preliminary",ascending=False).head(20)
    candidate_labels = [cache[(r.scenario,r.representation,r.algorithm,int(r.k))] for _,r in plausible_keys.iterrows() if (r.scenario,r.representation,r.algorithm,int(r.k)) in cache]
    bootstrap, consensus, obs_stability = full_bootstrap(preferred_x, algorithm, k, labels, df.country.astype(str).tolist(), n_boot, tables, candidate_labels)
    if len(bootstrap) and bootstrap.ARI.mean() < .60: warnings_list.append({"code":"bootstrap_instability","message":f"Mean bootstrap ARI is {bootstrap.ARI.mean():.3f}."})
    umap_coords = supplementary_umap(preferred_s.scaled, warnings_list)
    outliers, outlier_sens = outlier_analysis(preferred_x, df.country, preferred, labels, tables)
    all_neighbors, peers = peer_analysis(df, scenarios, tr, candidate_labels, tables, 20 if args.mode == "fast" else 100)
    profiles, domains, turkiye = cluster_profiles_and_turkiye(df, preferred_s, labels, tr, obs_stability, umap_coords, tables)
    if turkiye.iloc[0].membership_stability < .60: warnings_list.append({"code":"unstable_turkiye_membership","message":f"Türkiye membership stability is {turkiye.iloc[0].membership_stability:.3f}."})
    if peers.empty or peers.iloc[0].appearance_frequency < .50: warnings_list.append({"code":"unstable_peer_set","message":"No country appears in at least half of Türkiye top-10 neighbor configurations."})
    rank_profile = rank_profile_comparison(df, tr, preferred_x, peers, tables)
    tests = statistical_tests(pd.DataFrame(preferred_x, columns=[f"dimension_{i+1}" for i in range(preferred_x.shape[1])]), labels, tables)
    domain_del = domain_deletion(preferred_s, labels, algorithm, k, tables)
    candidate_row = candidates[(candidates.scenario==sid)&(candidates.representation==rep)&(candidates.algorithm==algorithm)&(candidates.k==k)].iloc[0]
    if pd.notna(candidate_row.get("cross_method_ARI")) and candidate_row.cross_method_ARI < 0.50:
        warnings_list.append({"code":"cross_method_disagreement","message":f"Preferred candidate is highly bootstrap-stable, but mean cross-method ARI is only {candidate_row.cross_method_ARI:.3f}; the partition is algorithm-dependent."})
    sensitivity = candidates.copy()
    sensitivity["bootstrap_ARI"] = sensitivity["bootstrap_ARI_preliminary"]
    sensitivity["bootstrap_AMI"] = sensitivity["bootstrap_AMI_preliminary"]
    preferred_mask = (sensitivity.scenario==sid)&(sensitivity.representation==rep)&(sensitivity.algorithm==algorithm)&(sensitivity.k==k)
    sensitivity.loc[preferred_mask,"bootstrap_ARI"] = bootstrap.ARI.mean() if len(bootstrap) else np.nan
    sensitivity.loc[preferred_mask,"bootstrap_AMI"] = bootstrap.AMI.mean() if len(bootstrap) else np.nan
    sensitivity["Türkiye_cluster"] = f"Cluster {labels[tr]+1}"
    sensitivity["Türkiye_membership_stability"] = obs_stability[tr]
    sensitivity["top_peer"] = peers.iloc[0].country if len(peers) else ""
    sensitivity["peer_stability"] = peers.iloc[0].appearance_frequency if len(peers) else np.nan
    required_cols=["scenario","representation","algorithm","k","silhouette","calinski_harabasz","davies_bouldin","cluster_sizes","imbalance_ratio","bootstrap_ARI","bootstrap_AMI","Türkiye_cluster","Türkiye_membership_stability","top_peer","peer_stability","candidate_flag"]
    save_csv(sensitivity[required_cols], tables/"sensitivity_summary.csv")
    figure_map = create_figures(df, preferred_s, labels, tr, validation, high_corr, consensus, outliers, domains, peers, rank_profile, candidates, umap_coords, figures)
    warnings_list.extend([{"code":"profile_interpretation","message":"Profile similarity != equal military power; geopolitical alignment; alliance; or combat equivalence."}])
    write_json(warnings_list, json_dir/"warnings.json")
    result = {
        "dataset": dataset,
        "feature_selection": {"primary_feature_count":len(features),"features":features},
        "missingness": {"total_missing":dataset["total_missing"],"naval_unknown_countries":int((naval.classification=="unknown_or_unreported").sum())},
        "transformations": {"log1p_feature_count":len(log_features),"features":sorted(log_features)},
        "pca": {"scenario":sid,"n_pc80":preferred_s.n_pc80,"n_pc90":preferred_s.n_pc90,"variance_pc2":float(np.cumsum(preferred_s.pca.explained_variance_ratio_)[min(1,len(preferred_s.pca.explained_variance_ratio_)-1)])},
        "cluster_validation": {"evaluated_rows":int(validation.silhouette.notna().sum()),"k_values":list(K_VALUES)},
        "preferred_solution": {"scenario":sid,"representation":rep,"algorithm":algorithm,"k":k,"selection_rule":"realism filter then lexicographic bootstrap ARI, cross-method ARI, silhouette, Davies-Bouldin","cluster_sizes":sizes.tolist()},
        "cluster_profiles": profiles.to_dict("records"),
        "stability": {"resamples_requested":n_boot,"resamples_successful":len(bootstrap),"mean_ARI":bootstrap.ARI.mean(),"mean_AMI":bootstrap.AMI.mean(),"mean_NMI":bootstrap.NMI.mean()},
        "consensus": {"mean_off_diagonal":float(consensus[np.triu_indices_from(consensus,1)].mean())},
        "outliers": outliers.head(10).to_dict("records"),
        "turkiye": turkiye.iloc[0].to_dict(),
        "turkiye_peers": peers.head(20).to_dict("records"),
        "rank_vs_profile": {"spearman":rank_profile.spearman_rank_difference_vs_profile_distance.iloc[0]},
        "statistical_tests": {"tests":len(tests),"fdr_significant":int((tests.get('q_value_bh',pd.Series(dtype=float))<.05).sum())},
        "sensitivity": {"domain_deletion":domain_del.to_dict("records"),"outlier_sensitivity":outlier_sens.to_dict("records")},
        "warnings": warnings_list,
        "figures": figure_map,
        "tables": {p.stem:str(p) for p in tables.glob("*.csv")},
    }
    write_json(result, json_dir/"manuscript_results.json")
    report = report_text(dataset,len(features),preferred,candidate_row,bootstrap,turkiye,peers,warnings_list)
    (reports/"capability_profile_analysis.md").write_text(report,encoding="utf-8")
    print(json.dumps({"mode":args.mode,"dataset_shape":[len(df),len(df.columns)],"primary_features":len(features),"preferred":preferred,"bootstrap_successful":len(bootstrap),"turkiye_cluster":int(labels[tr])+1,"turkiye_stability":float(obs_stability[tr]),"top_peer":peers.iloc[0].country if len(peers) else None,"output":str(root)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
