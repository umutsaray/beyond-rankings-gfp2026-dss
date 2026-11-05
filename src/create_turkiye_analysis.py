from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clean_gfp_dataset import clean_dataset
from extract_gfp_pdfs import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, RANK_FIELDS, build_outputs


PROJECT_TITLE = "Türkiye'nin Küresel Askerî Güç Konumunun Veri Analitiği ve Karşılaştırmalı Göstergelerle İncelenmesi"
MISSINGNESS_THRESHOLD = 0.20

RANK_FIELD_BY_METRIC = {
    "rank": "rank",
    "purchasing_power": "purchasing_power_rank",
    "foreign_reserves": "foreign_reserves_rank",
    "defense_budget": "defense_budget_rank",
    "external_debt": "external_debt_rank",
    "square_land_area": "square_land_area_rank",
    "coastline": "coastline_rank",
    "waterways": "waterways_rank",
    "total_population": "population_rank",
    "available_manpower": "available_manpower_rank",
    "fit_for_service": "fit_for_service_rank",
    "reaching_military_age_annually": "military_age_rank",
    "active_personnel": "active_personnel_rank",
    "reserve_personnel": "reserve_personnel_rank",
    "paramilitary_personnel": "paramilitary_rank",
    "aircraft_total": "aircraft_rank",
    "fighters": "fighters_rank",
    "attack_aircraft": "attack_aircraft_rank",
    "transports": "transports_rank",
    "trainers": "trainers_rank",
    "special_mission": "special_mission_rank",
    "tanker_fleet": "tankers_rank",
    "helicopters": "helicopters_rank",
    "attack_helicopters": "attack_helicopters_rank",
    "tanks": "tanks_rank",
    "vehicles": "vehicles_rank",
    "self_propelled_artillery": "artillery_sp_rank",
    "towed_artillery": "artillery_towed_rank",
    "mlrs": "mlrs_rank",
    "naval_total_assets": "ships_rank",
    "aircraft_carriers": "aircraft_carriers_rank",
    "helicopter_carriers": "helicopter_carriers_rank",
    "destroyers": "destroyers_rank",
    "frigates": "frigates_rank",
    "corvettes": "corvettes_rank",
    "submarines": "submarines_rank",
    "patrol_vessels": "patrol_vessels_rank",
    "mine_warfare": "mine_warfare_rank",
    "oil_proven_reserves": "oil_reserves_rank",
    "natural_gas_proven_reserves": "natural_gas_reserves_rank",
    "coal_proven_reserves": "coal_reserves_rank",
    "labor_force": "labor_force_rank",
    "merchant_marine_fleet": "merchant_marine_rank",
    "ports_trade_terminals": "ports_rank",
    "roadway_coverage": "roadways_rank",
    "railway_coverage": "railways_rank",
    "airports": "airports_rank",
}

INDICATOR_META = [
    {"metric": "rank", "label": "GFP sırası", "category": "Genel", "direction": "cost"},
    {"metric": "pwrindx", "label": "PwrIndx", "category": "Genel", "direction": "cost"},
    {"metric": "defense_budget", "label": "Savunma bütçesi", "category": "Finansal", "direction": "benefit"},
    {"metric": "purchasing_power", "label": "Satın alma gücü", "category": "Finansal", "direction": "benefit"},
    {"metric": "foreign_reserves", "label": "Döviz/altın rezervi", "category": "Finansal", "direction": "benefit"},
    {"metric": "square_land_area", "label": "Kara alanı", "category": "Coğrafya", "direction": "benefit"},
    {"metric": "coastline", "label": "Kıyı şeridi", "category": "Coğrafya", "direction": "benefit"},
    {"metric": "waterways", "label": "Su yolları", "category": "Coğrafya", "direction": "benefit"},
    {"metric": "total_population", "label": "Toplam nüfus", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "available_manpower", "label": "Kullanılabilir insan gücü", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "fit_for_service", "label": "Hizmete uygun nüfus", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "total_military_personnel", "label": "Toplam askerî personel", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "active_personnel", "label": "Aktif personel", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "reserve_personnel", "label": "Yedek personel", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "paramilitary_personnel", "label": "Paramiliter personel", "category": "İnsan gücü", "direction": "benefit"},
    {"metric": "aircraft_total", "label": "Toplam hava aracı", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "fighters", "label": "Avcı uçakları", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "attack_aircraft", "label": "Taarruz uçakları", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "transports", "label": "Nakliye uçakları", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "special_mission", "label": "Özel görev uçakları", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "tanker_fleet", "label": "Tanker filosu", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "helicopters", "label": "Helikopterler", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "attack_helicopters", "label": "Taarruz helikopterleri", "category": "Hava gücü", "direction": "benefit"},
    {"metric": "tanks", "label": "Tanklar", "category": "Kara gücü", "direction": "benefit"},
    {"metric": "vehicles", "label": "Zırhlı/askerî araçlar", "category": "Kara gücü", "direction": "benefit"},
    {"metric": "self_propelled_artillery", "label": "Kundağı motorlu topçu", "category": "Kara gücü", "direction": "benefit"},
    {"metric": "towed_artillery", "label": "Çekili topçu", "category": "Kara gücü", "direction": "benefit"},
    {"metric": "mlrs", "label": "ÇNRA/MLRS", "category": "Kara gücü", "direction": "benefit"},
    {"metric": "oil_proven_reserves", "label": "Kanıtlanmış petrol rezervi", "category": "Kaynak", "direction": "benefit"},
    {"metric": "natural_gas_proven_reserves", "label": "Kanıtlanmış doğal gaz rezervi", "category": "Kaynak", "direction": "benefit"},
    {"metric": "coal_proven_reserves", "label": "Kanıtlanmış kömür rezervi", "category": "Kaynak", "direction": "benefit"},
    {"metric": "labor_force", "label": "İşgücü", "category": "Lojistik", "direction": "benefit"},
    {"metric": "internet_coverage", "label": "İnternet kapsama oranı", "category": "Lojistik", "direction": "benefit"},
    {"metric": "merchant_marine_fleet", "label": "Ticaret filosu", "category": "Lojistik", "direction": "benefit"},
    {"metric": "ports_trade_terminals", "label": "Liman/ticaret terminali", "category": "Lojistik", "direction": "benefit"},
    {"metric": "airports", "label": "Havalimanları", "category": "Lojistik", "direction": "benefit"},
    {"metric": "roadway_coverage", "label": "Karayolu ağı", "category": "Lojistik", "direction": "benefit"},
    {"metric": "railway_coverage", "label": "Demiryolu ağı", "category": "Lojistik", "direction": "benefit"},
]

PROFILE_METRICS = [
    "rank",
    "pwrindx",
    "defense_budget",
    "purchasing_power",
    "foreign_reserves",
    "external_debt",
    "total_population",
    "available_manpower",
    "fit_for_service",
    "total_military_personnel",
    "active_personnel",
    "reserve_personnel",
    "paramilitary_personnel",
    "aircraft_total",
    "fighters",
    "tanker_fleet",
    "helicopters",
    "attack_helicopters",
    "tanks",
    "vehicles",
    "self_propelled_artillery",
    "towed_artillery",
    "mlrs",
    "naval_total_assets",
    "oil_proven_reserves",
    "natural_gas_proven_reserves",
    "coal_proven_reserves",
    "labor_force",
    "airports",
    "roadway_coverage",
    "railway_coverage",
]

COMPARISON_METRICS = [
    "rank",
    "pwrindx",
    "defense_budget",
    "purchasing_power",
    "active_personnel",
    "reserve_personnel",
    "aircraft_total",
    "fighters",
    "tanker_fleet",
    "attack_helicopters",
    "tanks",
    "vehicles",
    "self_propelled_artillery",
    "mlrs",
    "airports",
    "roadway_coverage",
    "railway_coverage",
]

RADAR_METRICS = [
    "defense_budget",
    "active_personnel",
    "reserve_personnel",
    "aircraft_total",
    "tanker_fleet",
    "attack_helicopters",
    "tanks",
    "vehicles",
]


def metric_lookup() -> dict[str, dict[str, str]]:
    return {item["metric"]: item for item in INDICATOR_META}


def ensure_clean_data(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    clean_path = output_dir / "tables" / "gfp_2026_clean.csv"
    raw_path = output_dir / "tables" / "gfp_2026_raw_extracted.csv"
    if clean_path.exists():
        print(f"Using existing clean dataset: {clean_path}")
        return pd.read_csv(clean_path)
    if raw_path.exists():
        print(f"Using existing raw extraction: {raw_path}")
        clean_dataset(output_dir)
        return pd.read_csv(clean_path)
    print(f"No local extraction found. Extracting PDFs from {input_dir}")
    build_outputs(input_dir, output_dir)
    clean_dataset(output_dir)
    return pd.read_csv(clean_path)


def find_turkiye(df: pd.DataFrame) -> pd.Series:
    mask = df["country"].astype(str).str.casefold().isin({"turkiye", "turkey", "türkiye"})
    if not mask.any():
        mask = df["source_file"].astype(str).str.contains("Turkiye|Turkey|Türkiye", case=False, na=False)
    if not mask.any():
        raise SystemExit("Could not find Turkiye in the cleaned dataset.")
    return df.loc[mask].iloc[0]


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric_df = df.copy()
    for column in columns:
        if column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
    return numeric_df


def selected_indicators(df: pd.DataFrame) -> list[dict[str, str]]:
    selected = []
    for item in INDICATOR_META:
        metric = item["metric"]
        if metric not in df.columns:
            continue
        missing_pct = pd.to_numeric(df[metric], errors="coerce").isna().mean()
        if missing_pct <= MISSINGNESS_THRESHOLD:
            selected.append(item)
    return selected


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:,.2f}".rstrip("0").rstrip(".")
    return f"{numeric:,.4f}".rstrip("0").rstrip(".")


def percentile_from_rank(rank: object, total: object) -> float:
    rank_value = pd.to_numeric(rank, errors="coerce")
    total_value = pd.to_numeric(total, errors="coerce")
    if pd.isna(rank_value) or pd.isna(total_value) or total_value <= 1:
        return np.nan
    return (1 - (rank_value - 1) / total_value) * 100


def relative_score(turkiye_value: float, reference_value: float, direction: str) -> float:
    if pd.isna(turkiye_value) or pd.isna(reference_value) or turkiye_value == 0 or reference_value == 0:
        return np.nan
    if direction == "cost":
        return reference_value / turkiye_value
    return turkiye_value / reference_value


def minmax_oriented_matrix(df: pd.DataFrame, indicators: list[dict[str, str]]) -> pd.DataFrame:
    metrics = [item["metric"] for item in indicators]
    matrix = df[metrics].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    oriented = pd.DataFrame(index=df.index)
    for item in indicators:
        metric = item["metric"]
        values = matrix[metric].astype(float)
        min_value = values.min()
        max_value = values.max()
        if np.isclose(max_value, min_value):
            oriented[metric] = 1.0
            continue
        if item["direction"] == "cost":
            oriented[metric] = (max_value - values) / (max_value - min_value)
        else:
            oriented[metric] = (values - min_value) / (max_value - min_value)
    return oriented


def entropy_weights(oriented: pd.DataFrame) -> pd.Series:
    eps = 1e-12
    values = oriented.clip(lower=0).to_numpy(dtype=float) + eps
    proportions = values / values.sum(axis=0, keepdims=True)
    n_rows = max(values.shape[0], 2)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(n_rows)
    diversification = 1 - entropy
    if np.isclose(diversification.sum(), 0):
        weights = np.repeat(1 / values.shape[1], values.shape[1])
    else:
        weights = diversification / diversification.sum()
    return pd.Series(weights, index=oriented.columns)


def critic_weights(oriented: pd.DataFrame) -> pd.Series:
    values = oriented.astype(float)
    std = values.std(axis=0, ddof=0)
    corr = values.corr().fillna(0)
    contrast = (1 - corr).sum(axis=0)
    information = std * contrast
    if np.isclose(information.sum(), 0):
        weights = np.repeat(1 / values.shape[1], values.shape[1])
    else:
        weights = information / information.sum()
    return pd.Series(weights, index=values.columns)


def topsis_scores(df: pd.DataFrame, indicators: list[dict[str, str]], weights: pd.Series) -> pd.Series:
    metrics = [item["metric"] for item in indicators]
    matrix = df[metrics].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).astype(float)
    denom = np.sqrt((matrix**2).sum(axis=0)).replace(0, 1)
    normalized = matrix / denom
    weighted = normalized * weights[metrics]

    ideal_best = {}
    ideal_worst = {}
    for item in indicators:
        metric = item["metric"]
        if item["direction"] == "cost":
            ideal_best[metric] = weighted[metric].min()
            ideal_worst[metric] = weighted[metric].max()
        else:
            ideal_best[metric] = weighted[metric].max()
            ideal_worst[metric] = weighted[metric].min()

    best = pd.Series(ideal_best)
    worst = pd.Series(ideal_worst)
    distance_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    return distance_worst / (distance_best + distance_worst)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_table1_dataset_summary(df: pd.DataFrame, turkiye: pd.Series, top10: pd.DataFrame, indicators: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for item in indicators:
        metric = item["metric"]
        values = pd.to_numeric(df[metric], errors="coerce")
        turkiye_value = pd.to_numeric(turkiye.get(metric), errors="coerce")
        top10_median = pd.to_numeric(top10[metric], errors="coerce").median()
        rows.append(
            {
                "indicator": metric,
                "label": item["label"],
                "category": item["category"],
                "direction": item["direction"],
                "country_count": int(values.notna().sum()),
                "missing_pct": round(float(values.isna().mean() * 100), 2),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "turkiye_value": turkiye_value,
                "top10_median": top10_median,
            }
        )
    return pd.DataFrame(rows)


def build_table2_turkiye_profile(df: pd.DataFrame, turkiye: pd.Series) -> pd.DataFrame:
    lookup = metric_lookup()
    total = turkiye.get("total_countries")
    rows = []
    for metric in PROFILE_METRICS:
        if metric not in df.columns:
            continue
        item = lookup.get(metric, {"label": metric.replace("_", " ").title(), "category": "Diğer", "direction": "benefit"})
        rank_field = RANK_FIELD_BY_METRIC.get(metric)
        global_rank = turkiye.get(rank_field) if rank_field in df.columns or rank_field == "rank" else np.nan
        rows.append(
            {
                "indicator": metric,
                "label": item["label"],
                "category": item["category"],
                "value": turkiye.get(metric),
                "global_rank": global_rank,
                "total_countries": total,
                "percentile_from_top": percentile_from_rank(global_rank, total),
            }
        )
    return pd.DataFrame(rows)


def build_table3_turkiye_vs_top10(turkiye: pd.Series, top10_reference: pd.DataFrame) -> pd.DataFrame:
    lookup = metric_lookup()
    rows = []
    for metric in COMPARISON_METRICS:
        if metric not in top10_reference.columns:
            continue
        item = lookup.get(metric)
        if not item:
            continue
        turkiye_value = pd.to_numeric(turkiye.get(metric), errors="coerce")
        top10_median = pd.to_numeric(top10_reference[metric], errors="coerce").median()
        rel = relative_score(turkiye_value, top10_median, item["direction"])
        rows.append(
            {
                "indicator": metric,
                "label": item["label"],
                "category": item["category"],
                "direction": item["direction"],
                "turkiye_value": turkiye_value,
                "top10_median": top10_median,
                "relative_to_top10_median": rel,
                "difference": turkiye_value - top10_median if pd.notna(turkiye_value) and pd.notna(top10_median) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("relative_to_top10_median", ascending=False)


def build_table4_rank_positions(turkiye: pd.Series) -> pd.DataFrame:
    total = turkiye.get("total_countries")
    rows = [
        {
            "rank_field": "rank",
            "metric": "overall_gfp_rank",
            "rank": pd.to_numeric(turkiye.get("rank"), errors="coerce"),
            "total_countries": total,
            "percentile_from_top": percentile_from_rank(turkiye.get("rank"), total),
        }
    ]
    for field in RANK_FIELDS:
        rank = pd.to_numeric(turkiye.get(field), errors="coerce")
        rows.append(
            {
                "rank_field": field,
                "metric": field.removesuffix("_rank"),
                "rank": rank,
                "total_countries": total,
                "percentile_from_top": percentile_from_rank(rank, total),
            }
        )
    return pd.DataFrame(rows).sort_values("rank", na_position="last")


def build_table5_strengths_weaknesses(table3: pd.DataFrame, table4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparison = table3.dropna(subset=["relative_to_top10_median"]).copy()
    strength_candidates = comparison[comparison["relative_to_top10_median"] >= 0.95].sort_values(
        "relative_to_top10_median", ascending=False
    )
    for _, row in strength_candidates.head(8).iterrows():
        interpretation = (
            "Top 10 medyanının üzerinde"
            if row["relative_to_top10_median"] >= 1
            else "Top 10 medyanına yakın konum"
        )
        rows.append(
            {
                "type": "strength",
                "indicator": row["indicator"],
                "label": row["label"],
                "basis": "Türkiye / Top 10 medyanı",
                "score": row["relative_to_top10_median"],
                "interpretation": interpretation,
            }
        )
    weakness_candidates = comparison[comparison["relative_to_top10_median"] < 0.8].sort_values("relative_to_top10_median")
    for _, row in weakness_candidates.head(8).iterrows():
        rows.append(
            {
                "type": "weakness",
                "indicator": row["indicator"],
                "label": row["label"],
                "basis": "Türkiye / Top 10 medyanı",
                "score": row["relative_to_top10_median"],
                "interpretation": "Top 10 medyanına göre geliştirme alanı",
            }
        )
    rank_strengths = table4.dropna(subset=["rank"]).head(5)
    for _, row in rank_strengths.iterrows():
        if row["rank"] <= 15:
            rows.append(
                {
                    "type": "strength",
                    "indicator": row["metric"],
                    "label": row["metric"].replace("_", " ").title(),
                    "basis": "Küresel sıra",
                    "score": row["rank"],
                    "interpretation": "Küresel sıralamada ilk 15 içinde",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["type", "indicator", "basis"])


def compute_multicriteria_tables(df: pd.DataFrame, indicators: list[dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    oriented = minmax_oriented_matrix(df, indicators)
    entropy = entropy_weights(oriented)
    critic = critic_weights(oriented)
    weights_df = pd.DataFrame(
        [
            {
                "indicator": item["metric"],
                "label": item["label"],
                "category": item["category"],
                "direction": item["direction"],
                "missing_pct": round(float(pd.to_numeric(df[item["metric"]], errors="coerce").isna().mean() * 100), 2),
                "entropy_weight": entropy[item["metric"]],
                "critic_weight": critic[item["metric"]],
                "average_weight": (entropy[item["metric"]] + critic[item["metric"]]) / 2,
            }
            for item in indicators
        ]
    ).sort_values("average_weight", ascending=False)

    entropy_score = topsis_scores(df, indicators, entropy)
    critic_score = topsis_scores(df, indicators, critic)
    ranking = pd.DataFrame(
        {
            "country": df["country"],
            "gfp_rank": pd.to_numeric(df["rank"], errors="coerce"),
            "pwrindx": pd.to_numeric(df["pwrindx"], errors="coerce"),
            "topsis_entropy_score": entropy_score,
            "topsis_critic_score": critic_score,
        }
    )
    ranking["topsis_average_score"] = ranking[["topsis_entropy_score", "topsis_critic_score"]].mean(axis=1)
    ranking["entropy_rank"] = ranking["topsis_entropy_score"].rank(ascending=False, method="min").astype(int)
    ranking["critic_rank"] = ranking["topsis_critic_score"].rank(ascending=False, method="min").astype(int)
    ranking["average_rank"] = ranking["topsis_average_score"].rank(ascending=False, method="min").astype(int)
    ranking = ranking.sort_values(["average_rank", "country"])
    return ranking, weights_df


def style_axis(ax: plt.Axes, title: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    sns.despine(ax=ax, left=True, bottom=False)


def save_top20_bar(df: pd.DataFrame, field: str, title: str, output_path: Path, ascending: bool = False, xlabel: str = "") -> None:
    plot_df = df[["country", field]].copy()
    plot_df[field] = pd.to_numeric(plot_df[field], errors="coerce")
    plot_df = plot_df.dropna(subset=[field]).sort_values(field, ascending=ascending).head(20)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=plot_df, x=field, y="country", hue="country", palette="crest", legend=False, ax=ax)
    style_axis(ax, title, xlabel or field.replace("_", " ").title())
    if ascending:
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_methodology_flowchart(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    steps = [
        ("PDF / mevcut veri", "gfp_pdfs veya outputs/tables"),
        ("Temizleme", "Birim dönüşümü ve eksik değer kontrolü"),
        ("Tanımlayıcı analiz", "Türkiye profili ve Top 10 medyanı"),
        ("Ağırlıklandırma", "Entropy ve CRITIC"),
        ("Sıralama", "TOPSIS yakınlık katsayısı"),
        ("Makale çıktıları", "Tablolar, figürler ve rapor"),
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for index, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):
        box = plt.Rectangle((x - 0.075, y - 0.12), 0.15, 0.24, facecolor="#e8f1f2", edgecolor="#1f4e5f", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y + 0.035, title, ha="center", va="center", fontsize=10, weight="bold", color="#173b45")
        ax.text(x, y - 0.045, textwrap.fill(subtitle, 18), ha="center", va="center", fontsize=8.5, color="#31515a")
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.085, y),
                xytext=(x + 0.085, y),
                arrowprops=dict(arrowstyle="->", color="#1f4e5f", lw=1.5),
            )
    ax.text(0.5, 0.88, "Yöntem Akışı", ha="center", va="center", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def normalized_for_radar(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    matrix = df[metrics].apply(pd.to_numeric, errors="coerce").fillna(0)
    matrix = np.log10(matrix.clip(lower=0) + 1)
    mins = matrix.min(axis=0)
    ranges = (matrix.max(axis=0) - mins).replace(0, 1)
    return (matrix - mins) / ranges


def save_radar(df: pd.DataFrame, top10_reference: pd.DataFrame, turkiye: pd.Series, output_path: Path) -> None:
    metrics = [metric for metric in RADAR_METRICS if metric in df.columns]
    radar_base = pd.concat([top10_reference, turkiye.to_frame().T], ignore_index=True)
    normalized = normalized_for_radar(radar_base, metrics)
    turkiye_values = normalized.iloc[-1].to_numpy()
    top10_median = normalized.iloc[:-1].median(axis=0).to_numpy()
    labels = [metric_lookup()[metric]["label"] for metric in metrics]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    turkiye_plot = np.concatenate([turkiye_values, [turkiye_values[0]]])
    top10_plot = np.concatenate([top10_median, [top10_median[0]]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, turkiye_plot, color="#0f766e", linewidth=2.5, label="Türkiye")
    ax.fill(angles, turkiye_plot, color="#0f766e", alpha=0.18)
    ax.plot(angles, top10_plot, color="#b45309", linewidth=2.2, linestyle="--", label="Top 10 medyanı")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Türkiye ve Top 10 Medyanı", pad=24, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_strength_weakness_bar(table3: pd.DataFrame, output_path: Path) -> None:
    plot_df = table3.dropna(subset=["relative_to_top10_median"]).copy()
    plot_df["difference_pct"] = (plot_df["relative_to_top10_median"] - 1) * 100
    plot_df = pd.concat([plot_df.nlargest(6, "difference_pct"), plot_df.nsmallest(6, "difference_pct")])
    plot_df = plot_df.drop_duplicates("indicator").sort_values("difference_pct")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = np.where(plot_df["difference_pct"] >= 0, "#0f766e", "#b91c1c")
    ax.barh(plot_df["label"], plot_df["difference_pct"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    style_axis(ax, "Türkiye'nin Top 10 Medyanına Göre Göreli Konumu", "Fark (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_topsis_top20(table6: pd.DataFrame, output_path: Path) -> None:
    plot_df = table6.sort_values("average_rank").head(20).sort_values("topsis_average_score")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["country"], plot_df["topsis_average_score"], color="#2563eb")
    style_axis(ax, "TOPSIS Bütünleşik Sıralama - İlk 20", "TOPSIS ortalama skor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(
    output_path: Path,
    df: pd.DataFrame,
    turkiye: pd.Series,
    table3: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
) -> None:
    turkiye_topsis = table6.loc[table6["country"].astype(str).str.casefold().eq(str(turkiye["country"]).casefold())].iloc[0]
    strengths = table5[table5["type"] == "strength"].head(5)
    weaknesses = table5[table5["type"] == "weakness"].head(5)
    top_weights = table7.head(7)
    top10 = table6.head(10)

    lines = [
        f"# {PROJECT_TITLE}",
        "",
        "## Veri Seti Özeti",
        (
            f"Çalışmada GFP 2026 raporlarından oluşturulan temiz veri seti kullanılmıştır. "
            f"Veri setinde {len(df)} ülke ve Türkiye odaklı karşılaştırmada kullanılan çoklu gösterge havuzu yer almaktadır. "
            "Yüksek eksik değer oranına sahip değişkenler, özellikle `naval_total_tonnage`, çok kriterli sıralama modelinden çıkarılmıştır."
        ),
        "",
        "## Türkiye Profili",
        (
            f"Türkiye, GFP genel sıralamasında {format_number(turkiye.get('rank'))}. sırada ve "
            f"PwrIndx değeri {format_number(turkiye.get('pwrindx'))} olan bir aktör olarak görünmektedir. "
            f"Savunma bütçesi {format_number(turkiye.get('defense_budget'))}, aktif personel sayısı "
            f"{format_number(turkiye.get('active_personnel'))}, toplam hava aracı {format_number(turkiye.get('aircraft_total'))} "
            f"ve tank kapasitesi {format_number(turkiye.get('tanks'))} olarak derlenmiştir."
        ),
        "",
        "## Türkiye - Top 10 Medyanı Karşılaştırması",
        (
            "Top 10 medyanı ile karşılaştırma, Türkiye'nin bazı kapasite göstergelerinde üst grup ile yakınsadığını, "
            "bazı finansal ve ölçek göstergelerinde ise farkın sürdüğünü göstermektedir. "
            "Özellikle kara gücü ve belirli hava gücü kalemleri Türkiye'nin göreli olarak güçlü olduğu alanlar arasında öne çıkmaktadır."
        ),
        "",
        "| Gösterge | Türkiye | Top 10 medyanı | Göreli skor |",
        "|---|---:|---:|---:|",
    ]
    for _, row in table3.head(8).iterrows():
        lines.append(
            f"| {row['label']} | {format_number(row['turkiye_value'])} | "
            f"{format_number(row['top10_median'])} | {format_number(row['relative_to_top10_median'])} |"
        )

    lines.extend(["", "## Güçlü ve Zayıf Yönler", ""])
    lines.append("Güçlü yönler:")
    for _, row in strengths.iterrows():
        lines.append(f"- {row['label']}: {row['interpretation']} (skor: {format_number(row['score'])}).")
    lines.append("")
    lines.append("Geliştirme alanları:")
    for _, row in weaknesses.iterrows():
        lines.append(f"- {row['label']}: {row['interpretation']} (skor: {format_number(row['score'])}).")

    lines.extend(
        [
            "",
            "## TOPSIS, Entropy ve CRITIC Bulguları",
            (
                f"Entropy ve CRITIC ağırlıkları ayrı ayrı hesaplanmış, ardından TOPSIS yakınlık katsayıları üretilmiştir. "
                f"Türkiye'nin Entropy tabanlı TOPSIS sırası {int(turkiye_topsis['entropy_rank'])}, "
                f"CRITIC tabanlı TOPSIS sırası {int(turkiye_topsis['critic_rank'])} ve ortalama TOPSIS sırası "
                f"{int(turkiye_topsis['average_rank'])} olarak hesaplanmıştır."
            ),
            "",
            "Ağırlıkların en yüksek olduğu göstergeler, ülkeler arasındaki ayrıştırıcı kapasitenin nerede yoğunlaştığını gösterir:",
        ]
    )
    for _, row in top_weights.iterrows():
        lines.append(f"- {row['label']}: ortalama ağırlık {format_number(row['average_weight'])}.")

    lines.extend(["", "TOPSIS ortalama skoruna göre ilk 10 ülke:"])
    for _, row in top10.iterrows():
        lines.append(f"- {int(row['average_rank'])}. {row['country']} ({format_number(row['topsis_average_score'])})")

    lines.extend(
        [
            "",
            "## Kısa Tartışma",
            (
                "Bulgular, Türkiye'nin yalnızca tekil envanter büyüklükleriyle değil, farklı kapasite alanlarının birlikte "
                "değerlendirildiği çok kriterli bir çerçevede de üst gruba yakın konumlandığını göstermektedir. "
                "Bununla birlikte finansal derinlik, stratejik kaynak rezervleri ve bazı lojistik ölçek göstergeleri, "
                "Top 10 medyanı ile kıyaslandığında politika ve kapasite geliştirme açısından izlenmesi gereken alanlardır."
            ),
            "",
            (
                "Bu analiz GFP verisinin ikincil ve açık kaynak niteliğine dayanır. Bu nedenle sonuçlar mutlak askerî etkinlik ölçümü "
                "olarak değil, göstergeler arası karşılaştırmalı konum analizi olarak okunmalıdır."
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_analysis(input_dir: Path, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    article_tables = output_dir / "article_tables"
    article_figures = output_dir / "article_figures"
    article_report = output_dir / "article_report"
    article_tables.mkdir(parents=True, exist_ok=True)
    article_figures.mkdir(parents=True, exist_ok=True)
    article_report.mkdir(parents=True, exist_ok=True)

    df = ensure_clean_data(input_dir, output_dir)
    numeric_columns = [item["metric"] for item in INDICATOR_META] + RANK_FIELDS + ["total_countries"]
    df = coerce_numeric(df, [column for column in numeric_columns if column in df.columns])
    turkiye = find_turkiye(df)

    ranked = df.dropna(subset=["rank"]).sort_values("rank").copy()
    top10_all = ranked.head(10)
    top10_reference = top10_all[~top10_all["country"].astype(str).str.casefold().eq(str(turkiye["country"]).casefold())]
    if top10_reference.empty:
        top10_reference = top10_all

    indicators = selected_indicators(df)
    table1 = build_table1_dataset_summary(df, turkiye, top10_reference, indicators)
    table2 = build_table2_turkiye_profile(df, turkiye)
    table3 = build_table3_turkiye_vs_top10(turkiye, top10_reference)
    table4 = build_table4_rank_positions(turkiye)
    table5 = build_table5_strengths_weaknesses(table3, table4)
    table6, table7 = compute_multicriteria_tables(df, indicators)

    save_csv(table1, article_tables / "table1_dataset_summary.csv")
    save_csv(table2, article_tables / "table2_turkiye_profile.csv")
    save_csv(table3, article_tables / "table3_turkiye_vs_top10.csv")
    save_csv(table4, article_tables / "table4_turkiye_global_rank_positions.csv")
    save_csv(table5, article_tables / "table5_turkiye_strengths_weaknesses.csv")
    save_csv(table6, article_tables / "table6_topsis_ranking.csv")
    save_csv(table7, article_tables / "table7_entropy_critic_weights.csv")

    save_methodology_flowchart(article_figures / "fig1_methodology_flowchart.png")
    save_top20_bar(ranked, "rank", "GFP Küresel Sıralama - İlk 20", article_figures / "fig2_top20_global_rank.png", ascending=True, xlabel="GFP sıra değeri (düşük daha iyi)")
    save_radar(df, top10_reference, turkiye, article_figures / "fig3_turkiye_vs_top10_radar.png")
    save_top20_bar(ranked, "defense_budget", "Savunma Bütçesi - İlk 20", article_figures / "fig4_top20_defense_budget.png", xlabel="USD")
    save_top20_bar(ranked, "tanks", "Tank Kapasitesi - İlk 20", article_figures / "fig5_top20_tanks.png", xlabel="Adet")
    save_strength_weakness_bar(table3, article_figures / "fig6_turkiye_strength_weakness_bar.png")
    save_topsis_top20(table6, article_figures / "fig7_topsis_top20.png")

    write_report(article_report / "turkiye_military_power_analysis_report.md", df, turkiye, table3, table5, table6, table7)

    try:
        from create_ml_analysis import create_ml_analysis

        create_ml_analysis(output_dir)
    except Exception as exc:
        error_path = article_report / "ml_analysis_error.txt"
        error_path.write_text(f"ML analysis did not complete:\n{exc}\n", encoding="utf-8")
        print(f"ML analysis did not complete. See {error_path}")

    try:
        from create_clustering_analysis import create_clustering_analysis

        create_clustering_analysis(output_dir)
    except Exception as exc:
        error_path = article_report / "clustering_analysis_error.txt"
        error_path.write_text(f"Clustering analysis did not complete:\n{exc}\n", encoding="utf-8")
        print(f"Clustering analysis did not complete. See {error_path}")

    try:
        from create_correlation_analysis import create_correlation_analysis

        create_correlation_analysis(output_dir)
    except Exception as exc:
        error_path = article_report / "correlation_analysis_error.txt"
        error_path.write_text(f"Correlation analysis did not complete:\n{exc}\n", encoding="utf-8")
        print(f"Correlation analysis did not complete. See {error_path}")

    print(f"Wrote article tables to {article_tables}")
    print(f"Wrote article figures to {article_figures}")
    print(f"Wrote report to {article_report / 'turkiye_military_power_analysis_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Türkiye-focused article tables, figures, and report.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing GFP PDF files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing generated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_analysis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
