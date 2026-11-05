from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from extract_gfp_pdfs import DEFAULT_OUTPUT_DIR


TARGET = "pwrindx"
ALPHA = 0.05
MISSINGNESS_THRESHOLD = 0.40
MIN_OBSERVATIONS = 20
CORRELATION_COLUMNS = [
    "feature",
    "label",
    "method",
    "n",
    "correlation",
    "p_value",
    "q_value",
    "abs_correlation",
    "relationship_with_power",
    "interpretation",
]

EXCLUDE_COLUMNS = {
    "country",
    "year",
    "source_file",
    "source_url",
    "rank",
    "total_countries",
    "last_reviewed_date",
    "pwrindx",
    "extraction_engine",
    "text_line_count",
    # Naval subtype fields contain GFP rank/sentinel artifacts in the extracted PDFs
    # (for example 145/999 values), so the cleaner aggregate naval_total_assets
    # is used instead for correlation analysis.
    "aircraft_carriers",
    "helicopter_carriers",
    "destroyers",
    "frigates",
    "corvettes",
    "submarines",
    "patrol_vessels",
    "mine_warfare",
}

FEATURE_LABELS = {
    "defense_budget": "Savunma bütçesi",
    "purchasing_power": "Satın alma gücü",
    "foreign_reserves": "Döviz/altın rezervi",
    "external_debt": "Dış borç",
    "square_land_area": "Kara alanı",
    "coastline": "Kıyı şeridi",
    "shared_borders": "Kara sınırları",
    "waterways": "Su yolları",
    "total_population": "Toplam nüfus",
    "available_manpower": "Kullanılabilir insan gücü",
    "fit_for_service": "Hizmete uygun nüfus",
    "reaching_military_age_annually": "Askerlik yaşına gelen nüfus",
    "total_military_personnel": "Toplam askerî personel",
    "active_personnel": "Aktif personel",
    "reserve_personnel": "Yedek personel",
    "paramilitary_personnel": "Paramiliter personel",
    "air_force_personnel": "Hava kuvvetleri personeli",
    "army_personnel": "Kara kuvvetleri personeli",
    "navy_personnel": "Deniz kuvvetleri personeli",
    "yearly_mobilization_potential": "Yıllık mobilizasyon potansiyeli",
    "aircraft_total": "Toplam hava aracı",
    "aircraft_readiness": "Hava aracı hazırlık",
    "fighters": "Avcı uçakları",
    "fighters_readiness": "Avcı hazırlık",
    "attack_aircraft": "Taarruz uçakları",
    "attack_aircraft_readiness": "Taarruz uçağı hazırlık",
    "transports": "Nakliye uçakları",
    "transports_readiness": "Nakliye uçağı hazırlık",
    "trainers": "Eğitim uçakları",
    "trainers_readiness": "Eğitim uçağı hazırlık",
    "special_mission": "Özel görev uçakları",
    "special_mission_readiness": "Özel görev hazırlık",
    "tanker_fleet": "Tanker filosu",
    "tanker_fleet_readiness": "Tanker hazırlık",
    "helicopters": "Helikopterler",
    "helicopters_readiness": "Helikopter hazırlık",
    "attack_helicopters": "Taarruz helikopterleri",
    "attack_helicopters_readiness": "Taarruz helikopteri hazırlık",
    "tanks": "Tanklar",
    "tanks_readiness": "Tank hazırlık",
    "vehicles": "Zırhlı/askerî araçlar",
    "vehicles_readiness": "Araç hazırlık",
    "self_propelled_artillery": "Kundağı motorlu topçu",
    "self_propelled_artillery_readiness": "Kundağı motorlu topçu hazırlık",
    "towed_artillery": "Çekili topçu",
    "towed_artillery_readiness": "Çekili topçu hazırlık",
    "mlrs": "ÇNRA/MLRS",
    "mlrs_readiness": "ÇNRA/MLRS hazırlık",
    "naval_total_assets": "Deniz toplam unsur",
    "aircraft_carriers": "Uçak gemileri",
    "helicopter_carriers": "Helikopter gemileri",
    "destroyers": "Destroyerler",
    "frigates": "Fırkateynler",
    "corvettes": "Korvetler",
    "submarines": "Denizaltılar",
    "patrol_vessels": "Devriye gemileri",
    "mine_warfare": "Mayın harbi unsurları",
    "oil_production": "Petrol üretimi",
    "oil_consumption": "Petrol tüketimi",
    "oil_deficit": "Petrol açığı/fazlası",
    "oil_proven_reserves": "Petrol rezervi",
    "natural_gas_production": "Doğal gaz üretimi",
    "natural_gas_consumption": "Doğal gaz tüketimi",
    "natural_gas_deficit": "Doğal gaz açığı/fazlası",
    "natural_gas_proven_reserves": "Doğal gaz rezervi",
    "coal_production": "Kömür üretimi",
    "coal_consumption": "Kömür tüketimi",
    "coal_deficit": "Kömür açığı/fazlası",
    "coal_proven_reserves": "Kömür rezervi",
    "internet_coverage": "İnternet kapsama oranı",
    "labor_force": "İşgücü",
    "merchant_marine_fleet": "Ticaret filosu",
    "ports_trade_terminals": "Liman/ticaret terminali",
    "airports": "Havalimanları",
    "roadway_coverage": "Karayolu ağı",
    "railway_coverage": "Demiryolu ağı",
}


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def load_dataset(output_dir: Path) -> pd.DataFrame:
    clean_path = output_dir / "tables" / "gfp_2026_clean.csv"
    if not clean_path.exists():
        raise SystemExit(f"Missing clean dataset: {clean_path}. Run clean_gfp_dataset.py first.")
    return pd.read_csv(clean_path)


def selected_features(df: pd.DataFrame) -> list[str]:
    features = []
    for column in df.columns:
        if column in EXCLUDE_COLUMNS or column.endswith("_rank"):
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() < MIN_OBSERVATIONS:
            continue
        if values.isna().mean() > MISSINGNESS_THRESHOLD:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        features.append(column)
    return features


def strength_direction(correlation: float) -> str:
    if correlation < 0:
        return "lower_pwrindx_stronger_power"
    return "higher_pwrindx_weaker_power"


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    n = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values = np.empty(n, dtype=float)
    q_values[order] = np.clip(adjusted, 0, 1)
    return q_values


def compute_correlations(df: pd.DataFrame, method: str) -> pd.DataFrame:
    target = pd.to_numeric(df[TARGET], errors="coerce")
    rows = []
    for feature in selected_features(df):
        values = pd.to_numeric(df[feature], errors="coerce")
        valid = target.notna() & values.notna()
        n = int(valid.sum())
        if n < MIN_OBSERVATIONS:
            continue
        if method == "pearson":
            coefficient, p_value = pearsonr(values[valid], target[valid])
        elif method == "spearman":
            coefficient, p_value = spearmanr(values[valid], target[valid])
        else:
            raise ValueError(f"Unsupported method: {method}")
        if pd.isna(coefficient) or pd.isna(p_value):
            continue
        rows.append(
            {
                "feature": feature,
                "label": label_for(feature),
                "method": method,
                "n": n,
                "correlation": coefficient,
                "p_value": p_value,
                "abs_correlation": abs(coefficient),
                "relationship_with_power": strength_direction(coefficient),
                "interpretation": "Daha yüksek gösterge değeri daha düşük PwrIndx ile ilişkilidir"
                if coefficient < 0
                else "Daha yüksek gösterge değeri daha yüksek PwrIndx ile ilişkilidir",
            }
        )
    result = pd.DataFrame(rows, columns=[column for column in CORRELATION_COLUMNS if column != "q_value"])
    if result.empty:
        return pd.DataFrame(columns=CORRELATION_COLUMNS)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    result = result[result["q_value"] < ALPHA].copy()
    if result.empty:
        return pd.DataFrame(columns=CORRELATION_COLUMNS)
    return result[CORRELATION_COLUMNS].sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def heatmap_features(pearson: pd.DataFrame, spearman: pd.DataFrame, limit: int = 18) -> list[str]:
    combined = pd.concat([pearson, spearman], ignore_index=True)
    if combined.empty:
        return []
    selected = (
        combined.groupby("feature", as_index=False)["abs_correlation"]
        .max()
        .sort_values("abs_correlation", ascending=False)
        .head(limit)["feature"]
        .tolist()
    )
    return [TARGET] + selected


def save_heatmap(df: pd.DataFrame, pearson: pd.DataFrame, spearman: pd.DataFrame, output_path: Path) -> None:
    features = heatmap_features(pearson, spearman)
    if len(features) < 3:
        return
    corr_df = df[features].apply(pd.to_numeric, errors="coerce")
    renamed = {feature: ("PwrIndx" if feature == TARGET else label_for(feature)) for feature in features}
    corr_df = corr_df.rename(columns=renamed)
    corr = corr_df.corr(method="spearman", min_periods=MIN_OBSERVATIONS)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title("PwrIndx ile İlişkili Göstergeler: Korelasyon Isı Haritası", fontsize=14, weight="bold", pad=14)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_top_correlations_figure(pearson: pd.DataFrame, spearman: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    merged = pearson[["feature", "label", "correlation", "p_value", "q_value"]].merge(
        spearman[["feature", "correlation", "p_value", "q_value"]],
        on="feature",
        how="outer",
        suffixes=("_pearson", "_spearman"),
    )
    if merged.empty:
        return merged
    spearman_labels = spearman.set_index("feature")["label"]
    merged["label"] = merged["label"].fillna(merged["feature"].map(spearman_labels))
    merged["plot_correlation"] = merged[["correlation_pearson", "correlation_spearman"]].mean(axis=1, skipna=True)
    merged["method_summary"] = np.select(
        [
            merged["correlation_pearson"].notna() & merged["correlation_spearman"].notna(),
            merged["correlation_pearson"].notna(),
        ],
        ["Pearson+Spearman", "Pearson"],
        default="Spearman",
    )
    negatives = merged[merged["plot_correlation"] < 0].nsmallest(10, "plot_correlation")
    positives = merged[merged["plot_correlation"] > 0].nlargest(10, "plot_correlation")
    plot_df = pd.concat([negatives, positives]).sort_values("plot_correlation")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = np.where(plot_df["plot_correlation"] < 0, "#0f766e", "#b91c1c")
    ax.barh(plot_df["label"], plot_df["plot_correlation"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title("PwrIndx ile En Güçlü Pozitif ve Negatif Korelasyonlar", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Anlamlı korelasyon katsayısı (iki yöntem varsa ortalama)\nnegatif: daha düşük PwrIndx, pozitif: daha yüksek PwrIndx")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return plot_df


def write_report(output_dir: Path, pearson: pd.DataFrame, spearman: pd.DataFrame, top_plot: pd.DataFrame) -> None:
    report_path = output_dir / "article_report" / "correlation_analysis_turkiye.md"
    strongest_lower = spearman[spearman["correlation"] < 0].sort_values("correlation").head(10)
    strongest_higher = spearman[spearman["correlation"] > 0].sort_values("correlation", ascending=False).head(10)

    lines = [
        "# Korelasyon Analizi: PwrIndx ile İlişkili Göstergeler",
        "",
        "## Yöntem",
        (
            "Analiz `outputs/tables/gfp_2026_clean.csv` dosyasındaki temiz GFP 2026 göstergelerine dayanır. "
            "Ülke adı, yıl, kaynak alanları, genel GFP sırası, toplam ülke sayısı, son inceleme tarihi, hedef değişken `pwrindx` "
            "ve sıralama türevi `_rank` alanları dışlanmıştır. %40'tan fazla eksik değere sahip değişkenler kullanılmamıştır. "
            "GFP sırası ya da sentinel değer karışımı içeren deniz alt kategori alanları dışlanmış, bunun yerine daha temiz "
            "`naval_total_assets` göstergesi korunmuştur. "
            f"Çoklu test etkisini sınırlamak için Benjamini-Hochberg FDR düzeltmesi uygulanmış ve yalnızca q<{ALPHA} "
            "düzeyinde istatistiksel olarak anlamlı ilişkiler raporlanmıştır."
        ),
        "",
        (
            "PwrIndx değerinde düşük skor daha güçlü askerî kapasiteye işaret eder. Bu nedenle negatif korelasyonlar, ilgili "
            "göstergenin artmasının daha düşük PwrIndx ve daha güçlü görünümle ilişkili olduğunu gösterir. Pozitif korelasyonlar ise "
            "göstergenin artmasının daha yüksek PwrIndx ile birlikte gözlendiğini ifade eder."
        ),
        "",
        f"Anlamlı Pearson ilişkisi sayısı: {len(pearson)}. Anlamlı Spearman ilişkisi sayısı: {len(spearman)}.",
        "",
        "## Daha Düşük PwrIndx ile En Güçlü İlişkili Faktörler",
    ]
    if strongest_lower.empty:
        lines.append("Anlamlı negatif ilişki bulunmamıştır.")
    else:
        for _, row in strongest_lower.iterrows():
            lines.append(
                f"- {row['label']}: Spearman ρ={row['correlation']:.3f}, "
                f"p={row['p_value']:.3g}, q={row['q_value']:.3g}, n={int(row['n'])}."
            )

    lines.extend(["", "## Daha Yüksek PwrIndx ile En Güçlü İlişkili Faktörler"])
    if strongest_higher.empty:
        lines.append("Anlamlı pozitif ilişki bulunmamıştır.")
    else:
        for _, row in strongest_higher.iterrows():
            lines.append(
                f"- {row['label']}: Spearman ρ={row['correlation']:.3f}, "
                f"p={row['p_value']:.3g}, q={row['q_value']:.3g}, n={int(row['n'])}."
            )

    lines.extend(
        [
            "",
            "## Türkiye Odaklı Yorum",
            (
                "Korelasyon sonuçları, Türkiye'nin hangi tekil göstergelerde güçlü ya da zayıf olduğunu doğrudan kanıtlamaz; "
                "ülkeler arası gözlemsel ilişkileri gösterir. Bununla birlikte negatif yöndeki güçlü ve anlamlı ilişkiler, "
                "PwrIndx'i düşüren yani askerî güç görünümünü iyileştiren kapasite alanlarının genel örüntüsünü ortaya koyar. "
                "Pozitif yöndeki anlamlı ilişkiler ise veri setinde daha yüksek PwrIndx ile birlikte değişen göstergeleri gösterir ve "
                "nedensel bir sonuç olarak okunmamalıdır."
            ),
            "",
            "Bu analiz yalnızca mevcut GFP 2026 göstergelerine dayalı, istatistiksel ve gösterge-temelli bir değerlendirmedir.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def update_main_report(output_dir: Path, pearson: pd.DataFrame, spearman: pd.DataFrame) -> None:
    report_path = output_dir / "article_report" / "turkiye_military_power_analysis_report.md"
    if not report_path.exists():
        return
    marker = "## Korelasyon Bulguları"
    text = report_path.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker)[0].rstrip()
    strongest = spearman[spearman["correlation"] < 0].sort_values("correlation").head(3)
    factors = ", ".join(strongest["label"].tolist()) if not strongest.empty else "anlamlı negatif ilişki bulunmamıştır"
    section = (
        f"\n\n{marker}\n"
        f"Pearson ve Spearman korelasyon analizlerinde yalnızca FDR düzeltilmiş q<{ALPHA} düzeyindeki anlamlı ilişkiler raporlanmıştır. "
        f"Daha düşük PwrIndx ile en güçlü ilişkili faktörler arasında {factors} öne çıkmaktadır. "
        "Negatif korelasyonlar PwrIndx'i düşüren, yani daha güçlü askerî görünümle birlikte hareket eden göstergeler olarak yorumlanmalıdır; "
        "bu bulgular nedensel değil, gösterge-temelli istatistiksel ilişkiler olarak okunmalıdır."
    )
    report_path.write_text(text + section + "\n", encoding="utf-8")


def create_correlation_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    table_dir = output_dir / "article_tables"
    figure_dir = output_dir / "article_figures"
    report_dir = output_dir / "article_report"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(output_dir)
    pearson = compute_correlations(df, "pearson")
    spearman = compute_correlations(df, "spearman")

    save_csv(pearson, table_dir / "table14_pearson_correlations_with_pwrindx.csv")
    save_csv(spearman, table_dir / "table15_spearman_correlations_with_pwrindx.csv")

    save_heatmap(df, pearson, spearman, figure_dir / "fig17_correlation_heatmap.png")
    top_plot = save_top_correlations_figure(pearson, spearman, figure_dir / "fig18_top_positive_negative_correlations.png")

    write_report(output_dir, pearson, spearman, top_plot)
    update_main_report(output_dir, pearson, spearman)

    print(f"Wrote correlation tables to {table_dir}")
    print(f"Wrote correlation figures to {figure_dir}")
    print(f"Wrote correlation report to {report_dir / 'correlation_analysis_turkiye.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pwrindx correlation analysis for Türkiye-focused GFP article.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing generated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_correlation_analysis(args.output_dir)


if __name__ == "__main__":
    main()
