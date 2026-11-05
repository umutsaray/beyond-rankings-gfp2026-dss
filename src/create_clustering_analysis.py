from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from extract_gfp_pdfs import DEFAULT_OUTPUT_DIR


RANDOM_STATE = 42
MISSINGNESS_THRESHOLD = 0.40
K_RANGE = range(2, 9)

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
}

PROFILE_COLUMNS = [
    "rank",
    "pwrindx",
    "defense_budget",
    "active_personnel",
    "aircraft_total",
    "tanks",
    "naval_total_assets",
    "purchasing_power",
]

RADAR_METRICS = [
    "defense_budget",
    "active_personnel",
    "aircraft_total",
    "tanks",
    "naval_total_assets",
    "purchasing_power",
]

FEATURE_LABELS = {
    "defense_budget": "Savunma bütçesi",
    "active_personnel": "Aktif personel",
    "aircraft_total": "Toplam hava aracı",
    "tanks": "Tanklar",
    "naval_total_assets": "Deniz unsurları",
    "purchasing_power": "Satın alma gücü",
    "rank": "GFP sırası",
    "pwrindx": "PwrIndx",
}


def label_for(metric: str) -> str:
    return FEATURE_LABELS.get(metric, metric.replace("_", " ").title())


def load_dataset(output_dir: Path) -> pd.DataFrame:
    clean_path = output_dir / "tables" / "gfp_2026_clean.csv"
    if not clean_path.exists():
        raise SystemExit(f"Missing clean dataset: {clean_path}. Run clean_gfp_dataset.py first.")
    return pd.read_csv(clean_path)


def find_turkiye_index(df: pd.DataFrame) -> int:
    mask = df["country"].astype(str).str.casefold().isin({"turkiye", "turkey", "türkiye"})
    if not mask.any():
        mask = df["source_file"].astype(str).str.contains("Turkiye|Turkey|Türkiye", case=False, na=False)
    if not mask.any():
        raise SystemExit("Could not find Turkiye in the cleaned dataset.")
    return int(df.index[mask][0])


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    feature_columns = []
    audit_rows = []
    for column in df.columns:
        if column in EXCLUDE_COLUMNS or column.endswith("_rank"):
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        missing_pct = values.isna().mean()
        selected = missing_pct <= MISSINGNESS_THRESHOLD
        audit_rows.append(
            {
                "feature": column,
                "missing_pct": round(float(missing_pct * 100), 2),
                "selected": selected,
            }
        )
        if selected:
            feature_columns.append(column)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(X)
    scaled = StandardScaler().fit_transform(imputed)
    audit = pd.DataFrame(audit_rows).sort_values(["selected", "missing_pct", "feature"], ascending=[False, True, True])
    return X, scaled, feature_columns, audit


def elbow_k(k_values: list[int], inertias: list[float]) -> int:
    points = np.column_stack([k_values, inertias]).astype(float)
    start = points[0]
    end = points[-1]
    line = end - start
    if np.allclose(line, 0):
        return int(k_values[0])
    deltas = start - points
    distances = np.abs(line[0] * deltas[:, 1] - line[1] * deltas[:, 0]) / np.linalg.norm(line)
    return int(k_values[int(np.argmax(distances))])


def evaluate_k_values(scaled: np.ndarray) -> tuple[pd.DataFrame, int, int]:
    rows = []
    k_values = []
    inertias = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(scaled)
        inertia = float(model.inertia_)
        silhouette = float(silhouette_score(scaled, labels))
        rows.append({"k": k, "inertia": inertia, "silhouette_score": silhouette})
        k_values.append(k)
        inertias.append(inertia)
    selection = pd.DataFrame(rows)
    selected_k = int(selection.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"])
    elbow = elbow_k(k_values, inertias)
    return selection, selected_k, elbow


def assign_clusters(scaled: np.ndarray, selected_k: int) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=50)
    kmeans_labels = kmeans.fit_predict(scaled)
    hierarchical = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(scaled)
    return kmeans_labels, hierarchical_labels


def pca_coordinates(scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    return coords, pca


def cluster_interpretation_labels(profile: pd.DataFrame) -> dict[int, str]:
    ordered = profile.sort_values(["median_pwrindx", "median_rank"], na_position="last")["cluster_id"].tolist()
    n_clusters = len(ordered)
    if n_clusters == 2:
        labels = ["Super / major powers", "Regional, medium and limited capacity countries"]
    elif n_clusters == 3:
        labels = ["Super powers", "Major / regional powers", "Medium and limited capacity countries"]
    elif n_clusters == 4:
        labels = ["Super powers", "Major powers", "Regional powers", "Medium and limited capacity countries"]
    else:
        labels = ["Super powers", "Major powers", "Regional powers", "Medium powers", "Limited capacity countries"]
        labels.extend([f"Limited capacity countries - subgroup {i}" for i in range(2, n_clusters - 3)])
    return {cluster_id: labels[index] for index, cluster_id in enumerate(ordered)}


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby("kmeans_cluster"):
        row = {"cluster_id": int(cluster_id), "country_count": int(len(group))}
        for column in PROFILE_COLUMNS:
            row[f"median_{column}"] = pd.to_numeric(group[column], errors="coerce").median()
        rows.append(row)
    profile = pd.DataFrame(rows)
    labels = cluster_interpretation_labels(profile)
    profile["interpretation_label"] = profile["cluster_id"].map(labels)
    return profile.sort_values(["median_pwrindx", "median_rank"], na_position="last")


def build_neighbors(df: pd.DataFrame, scaled: np.ndarray, turkiye_idx: int) -> pd.DataFrame:
    turkiye_cluster = int(df.loc[turkiye_idx, "kmeans_cluster"])
    distances = np.linalg.norm(scaled - scaled[list(df.index).index(turkiye_idx)], axis=1)
    neighbor_df = df.loc[df["kmeans_cluster"] == turkiye_cluster].copy()
    neighbor_df["distance_to_turkiye"] = distances[neighbor_df.index]
    columns = [
        "country",
        "rank",
        "pwrindx",
        "kmeans_cluster",
        "hierarchical_cluster",
        "pca1",
        "pca2",
        "distance_to_turkiye",
        "defense_budget",
        "active_personnel",
        "aircraft_total",
        "tanks",
        "naval_total_assets",
    ]
    existing = [column for column in columns if column in neighbor_df.columns]
    return neighbor_df[existing].sort_values(["pwrindx", "rank"], na_position="last")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_pca_plot(df: pd.DataFrame, cluster_column: str, title: str, output_path: Path, turkiye_idx: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=df[cluster_column].nunique())
    sns.scatterplot(
        data=df,
        x="pca1",
        y="pca2",
        hue=cluster_column,
        palette=palette,
        s=55,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    for cluster_id, group in df.groupby(cluster_column):
        ax.text(
            group["pca1"].median(),
            group["pca2"].median(),
            f"C{int(cluster_id)}",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#64748b", alpha=0.88),
        )

    tr = df.loc[turkiye_idx]
    ax.scatter([tr["pca1"]], [tr["pca2"]], s=230, marker="*", color="#dc2626", edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(
        "Türkiye",
        xy=(tr["pca1"], tr["pca2"]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=12,
        weight="bold",
        color="#991b1b",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
    )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    ax.legend(title="Cluster", loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_silhouette_plot(selection: pd.DataFrame, selected_k: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["silhouette_score"], marker="o", color="#0f766e", linewidth=2)
    ax.axvline(selected_k, color="#dc2626", linestyle="--", label=f"Selected k={selected_k}")
    ax.set_title("Silhouette Scores by Cluster Count", fontsize=13, weight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(selection["k"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_elbow_plot(selection: pd.DataFrame, elbow: int, selected_k: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["inertia"], marker="o", color="#2563eb", linewidth=2)
    ax.axvline(elbow, color="#f59e0b", linestyle="--", label=f"Elbow k={elbow}")
    ax.axvline(selected_k, color="#dc2626", linestyle=":", label=f"Silhouette k={selected_k}")
    ax.set_title("Elbow Curve", fontsize=13, weight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Within-cluster inertia")
    ax.set_xticks(selection["k"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def normalize_radar_values(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    values = df[metrics].apply(pd.to_numeric, errors="coerce").fillna(df[metrics].median(numeric_only=True))
    values = np.log10(values.clip(lower=0) + 1)
    mins = values.min(axis=0)
    ranges = (values.max(axis=0) - mins).replace(0, 1)
    return (values - mins) / ranges


def save_cluster_radar(df: pd.DataFrame, turkiye_idx: int, output_path: Path) -> None:
    metrics = [metric for metric in RADAR_METRICS if metric in df.columns]
    turkiye_cluster = int(df.loc[turkiye_idx, "kmeans_cluster"])
    cluster_median = df[df["kmeans_cluster"] == turkiye_cluster][metrics].median(numeric_only=True)
    global_median = df[metrics].median(numeric_only=True)
    radar_base = pd.concat([df.loc[[turkiye_idx], metrics], cluster_median.to_frame().T, global_median.to_frame().T], ignore_index=True)
    normalized = normalize_radar_values(pd.concat([df[metrics], radar_base], ignore_index=True), metrics).tail(3)

    labels = [label_for(metric) for metric in metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    series = {
        "Türkiye": normalized.iloc[0].to_numpy(),
        "Türkiye küme medyanı": normalized.iloc[1].to_numpy(),
        "Küresel medyan": normalized.iloc[2].to_numpy(),
    }
    colors = {"Türkiye": "#dc2626", "Türkiye küme medyanı": "#2563eb", "Küresel medyan": "#64748b"}

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, polar=True)
    for name, values in series.items():
        plot_values = np.concatenate([values, [values[0]]])
        ax.plot(angles, plot_values, linewidth=2.2, label=name, color=colors[name])
        ax.fill(angles, plot_values, alpha=0.10, color=colors[name])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Türkiye Küme Profili", fontsize=14, weight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def nearest_countries_by_distance(neighbors: pd.DataFrame) -> pd.DataFrame:
    return neighbors[neighbors["country"].astype(str).str.casefold() != "turkiye"].sort_values("distance_to_turkiye").head(10)


def write_clustering_report(
    output_dir: Path,
    df: pd.DataFrame,
    selection: pd.DataFrame,
    selected_k: int,
    elbow: int,
    profiles: pd.DataFrame,
    neighbors: pd.DataFrame,
    turkiye_idx: int,
    feature_count: int,
) -> None:
    report_path = output_dir / "article_report" / "clustering_interpretation_turkiye.md"
    turkiye = df.loc[turkiye_idx]
    turkiye_cluster = int(turkiye["kmeans_cluster"])
    profile = profiles[profiles["cluster_id"] == turkiye_cluster].iloc[0]
    close_neighbors = nearest_countries_by_distance(neighbors)
    topsis_path = output_dir / "article_tables" / "table6_topsis_ranking.csv"
    topsis_sentence = "TOPSIS tablosu bulunmadığı için bu bölümde TOPSIS karşılaştırması yapılamamıştır."
    if topsis_path.exists():
        topsis = pd.read_csv(topsis_path)
        tr_topsis = topsis[topsis["country"].astype(str).str.casefold().eq(str(turkiye["country"]).casefold())]
        if not tr_topsis.empty:
            topsis_sentence = (
                f"TOPSIS ortalama sıralamasında Türkiye {int(tr_topsis.iloc[0]['average_rank'])}. sıradadır. "
                "Kümeleme ise sıralama üretmekten çok benzer gösterge profillerini grupladığı için, TOPSIS sonucu ile birebir aynı "
                "bir hiyerarşi beklenmemelidir."
            )

    lines = [
        "# Kümeleme ve Boyut İndirgeme Analizi: Türkiye'nin Askerî Kapasite Kümesi",
        "",
        "## 1. Kümeleme Metodolojisi",
        (
            f"Analiz, `outputs/tables/gfp_2026_clean.csv` dosyasındaki GFP 2026 göstergelerine dayanmaktadır. "
            f"Ülke adı, yıl, kaynak alanları, GFP genel sırası, toplam ülke sayısı, son inceleme tarihi ve hedef nitelikteki "
            f"`pwrindx` alanı kümeleme özelliklerinden çıkarılmıştır. Ayrıca doğrudan sıralama türevi olan `_rank` alanları "
            f"kullanılmamış ve %40'tan fazla eksik değere sahip değişkenler dışlanmıştır. Kalan {feature_count} sayısal gösterge "
            "medyan ile tamamlanmış ve standartlaştırılmıştır."
        ),
        "",
        "PCA yalnızca iki boyutlu görselleştirme için kullanılmıştır; K-Means ve hiyerarşik aglomeratif kümeleme standartlaştırılmış gösterge matrisi üzerinde uygulanmıştır.",
        "",
        "## 2. Küme Sayısı Seçimi",
        (
            f"k=2..8 aralığı test edilmiştir. En yüksek silhouette skoru k={selected_k} için elde edilmiştir. "
            f"Elbow yöntemi ise k={elbow} değerini işaret etmektedir. Kullanıcı kuralına uygun olarak silhouette tercih edilmiş "
            f"ve nihai küme atamaları k={selected_k} ile yapılmıştır."
        ),
        "",
        "| k | Silhouette | Inertia |",
        "|---:|---:|---:|",
    ]
    for _, row in selection.iterrows():
        lines.append(f"| {int(row['k'])} | {row['silhouette_score']:.4f} | {row['inertia']:.2f} |")

    lines.extend(
        [
            "",
            "## 3. Türkiye'nin Kümesi",
            (
                f"Türkiye, K-Means sonucunda C{turkiye_cluster} kümesinde yer almaktadır. Bu küme için yorum etiketi "
                f"**{profile['interpretation_label']}** olarak atanmıştır. Kümenin ülke sayısı {int(profile['country_count'])}, "
                f"medyan GFP sırası {profile['median_rank']:.1f} ve medyan PwrIndx değeri {profile['median_pwrindx']:.4f} düzeyindedir."
            ),
            "",
            "Bu sonuç, Türkiye'nin GFP genel sıralamasındaki üst konumunu doğrudan yeniden üretmek yerine, çok sayıda kapasite göstergesinin ortak desenine göre hangi ülke grubuyla benzeştiğini gösterir.",
            "",
            "## 4. Türkiye'ye En Yakın Ülkeler",
            "Aşağıdaki ülkeler, Türkiye ile aynı K-Means kümesinde yer alan ve standartlaştırılmış gösterge uzayında Türkiye'ye en yakın gözlenen ülkelerdir:",
        ]
    )
    for _, row in close_neighbors.iterrows():
        lines.append(f"- {row['country']} (GFP sıra: {int(row['rank'])}, PwrIndx: {row['pwrindx']:.4f})")

    lines.extend(
        [
            "",
            "## 5. Küme Düzeyi Yorum",
            (
                "Yorum etiketleri veri odaklıdır ve jeopolitik hüküm olarak okunmamalıdır. Seçilen k değeri iki olduğunda, "
                "silhouette metriği en belirgin ayrımın yüksek kapasiteli ülkeler ile daha geniş ülke grubu arasında olduğunu göstermektedir. "
                "Bu nedenle klasik beşli ifade şu şekilde okunmalıdır: super powers, major powers, regional powers, medium powers ve limited "
                "capacity countries kategorileri k değerine bağlı olarak birleşik kümeler hâlinde temsil edilebilir."
            ),
            "",
        ]
    )
    for _, row in profiles.iterrows():
        lines.append(
            f"- C{int(row['cluster_id'])} ({row['interpretation_label']}): {int(row['country_count'])} ülke, "
            f"medyan sıra {row['median_rank']:.1f}, medyan PwrIndx {row['median_pwrindx']:.4f}."
        )

    lines.extend(
        [
            "",
            "## 6. GFP ve TOPSIS ile İlişki",
            (
                f"Türkiye'nin GFP genel sırası {int(turkiye['rank'])} ve PwrIndx değeri {turkiye['pwrindx']:.4f}'tür. "
                f"{topsis_sentence} Kümeleme, özellikle kapasite büyüklüğü, insan gücü, hava/kara unsurları, lojistik ve kaynak göstergelerinin "
                "birlikte ürettiği benzerlik yapısını yakalar."
            ),
            "",
            "Bu nedenle sonuçlar, yalnızca mevcut GFP 2026 göstergelerine dayalı veri analitiği bulguları olarak yorumlanmalıdır; mutlak askerî etkinlik veya operasyonel üstünlük iddiası taşımaz.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def update_main_article_report(output_dir: Path, selected_k: int, elbow: int, turkiye_cluster: int, cluster_label: str) -> None:
    report_path = output_dir / "article_report" / "turkiye_military_power_analysis_report.md"
    if not report_path.exists():
        return
    marker = "## Kümeleme ve Boyut İndirgeme Bulguları"
    text = report_path.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker)[0].rstrip()
    section = (
        f"\n\n{marker}\n"
        f"Kümeleme analizinde k=2..8 aralığı silhouette ve elbow ölçütleriyle test edilmiştir. "
        f"Silhouette ölçütü k={selected_k} değerini, elbow yöntemi ise k={elbow} değerini işaret etmiştir; nihai atamada silhouette sonucu tercih edilmiştir. "
        f"Türkiye K-Means sonucunda C{turkiye_cluster} kümesinde yer almakta ve bu küme `{cluster_label}` olarak yorumlanmaktadır. "
        "Bu sonuç, GFP ve TOPSIS sıralamalarını tamamlayan gösterge-temelli bir benzerlik analizi olarak değerlendirilmelidir."
    )
    report_path.write_text(text + section + "\n", encoding="utf-8")


def create_clustering_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    table_dir = output_dir / "article_tables"
    figure_dir = output_dir / "article_figures"
    report_dir = output_dir / "article_report"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    source = load_dataset(output_dir)
    X, scaled, feature_columns, audit = prepare_feature_matrix(source)
    selection, selected_k, elbow = evaluate_k_values(scaled)
    kmeans_labels, hierarchical_labels = assign_clusters(scaled, selected_k)
    coords, pca = pca_coordinates(scaled)

    result = source.copy()
    result["kmeans_cluster"] = kmeans_labels
    result["hierarchical_cluster"] = hierarchical_labels
    result["pca1"] = coords[:, 0]
    result["pca2"] = coords[:, 1]

    turkiye_idx = find_turkiye_index(result)
    profiles = build_cluster_profiles(result)
    neighbors = build_neighbors(result, scaled, turkiye_idx)
    turkiye_cluster = int(result.loc[turkiye_idx, "kmeans_cluster"])
    turkiye_label = str(profiles.loc[profiles["cluster_id"] == turkiye_cluster, "interpretation_label"].iloc[0])

    table11 = result[["country", "rank", "pwrindx", "kmeans_cluster", "hierarchical_cluster", "pca1", "pca2"]].sort_values(["kmeans_cluster", "rank"])
    save_csv(table11, table_dir / "table11_cluster_assignments.csv")
    save_csv(profiles, table_dir / "table12_cluster_profiles.csv")
    save_csv(neighbors, table_dir / "table13_turkiye_cluster_neighbors.csv")

    save_csv(selection, report_dir / "clustering_k_selection.csv")
    save_csv(audit, report_dir / "clustering_feature_audit.csv")

    save_pca_plot(result, "kmeans_cluster", "PCA Visualization - K-Means Clusters", figure_dir / "fig12_pca_kmeans_clusters.png", turkiye_idx)
    save_pca_plot(result, "hierarchical_cluster", "PCA Visualization - Hierarchical Clusters", figure_dir / "fig13_pca_hierarchical_clusters.png", turkiye_idx)
    save_silhouette_plot(selection, selected_k, figure_dir / "fig14_silhouette_scores.png")
    save_elbow_plot(selection, elbow, selected_k, figure_dir / "fig15_elbow_curve.png")
    save_cluster_radar(result, turkiye_idx, figure_dir / "fig16_turkiye_cluster_profile_radar.png")

    write_clustering_report(output_dir, result, selection, selected_k, elbow, profiles, neighbors, turkiye_idx, len(feature_columns))
    update_main_article_report(output_dir, selected_k, elbow, turkiye_cluster, turkiye_label)

    print(f"Wrote clustering tables to {table_dir}")
    print(f"Wrote clustering figures to {figure_dir}")
    print(f"Wrote clustering report to {report_dir / 'clustering_interpretation_turkiye.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clustering and PCA analysis for Türkiye-focused GFP article.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing generated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_clustering_analysis(args.output_dir)


if __name__ == "__main__":
    main()
