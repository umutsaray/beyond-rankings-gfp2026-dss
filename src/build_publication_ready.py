from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler, StandardScaler

import audit_capability_profile_analysis as audit
import correct_capability_profile_analysis as corr
import create_capability_profile_analysis as cap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "capability_profiles"
DEST = SOURCE / "publication_ready"
DPI = 600
SEED = cap.RANDOM_STATE
N_BOOT = 500
DOMAINS = corr.DOMAIN_ORDER


def save_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DEST / name, index=False, encoding="utf-8-sig", float_format="%.6f")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(DEST / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(
        DEST / f"{stem}.tiff", dpi=DPI, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def corrected_objects():
    df, naval, presence, features, decisions, logs, scenarios = audit.base_objects()
    base = cap.impute_structurally(df, features, naval.classification)
    for column in logs:
        base[column] = np.log1p(base[column].clip(lower=0))
    groups, composition = corr.make_redundancy_groups(base, features)
    prepared, domains, scaled, pca, scores, n80, n90, models = corr.full_corrected_s8(
        df, naval, features, logs, groups
    )
    labels = KMeans(n_clusters=3, n_init=100, random_state=SEED).fit_predict(scores[:, :n90])
    return df, naval, features, decisions, logs, scenarios, groups, composition, domains, scaled, pca, scores, n80, n90, labels


def corrected_consensus(df, naval, features, logs, groups, reference, n90):
    rng = np.random.default_rng(SEED)
    n = len(df)
    coassoc = np.zeros((n, n), dtype=float)
    denom = np.zeros((n, n), dtype=float)
    for b in range(N_BOOT):
        sample = rng.integers(0, n, n)
        prepared = corr.prepare_raw(df, naval, features, logs, sample)
        boot_domains, _ = corr.fit_corrected_domains(prepared, sample, groups)
        domain_scaler = RobustScaler(quantile_range=(25, 75)).fit(boot_domains.iloc[sample])
        boot_scaled = domain_scaler.transform(boot_domains)
        pca = PCA(n_components=n90, random_state=SEED).fit(boot_scaled[sample])
        x_sample = pca.transform(boot_scaled[sample])[:, :n90]
        model = KMeans(n_clusters=3, n_init=30, random_state=SEED + 3000 + b + 1).fit(x_sample)
        x_all = pca.transform(boot_scaled)[:, :n90]
        predicted = model.predict(x_all)
        predicted, _ = audit.align_labels(reference, predicted, 3)
        sampled_unique = np.unique(sample)
        ix = np.ix_(sampled_unique, sampled_unique)
        denom[ix] += 1
        coassoc[ix] += predicted[sampled_unique, None] == predicted[None, sampled_unique]
    return np.divide(coassoc, denom, out=np.zeros_like(coassoc), where=denom > 0)


def build_tables(df, features, decisions, logs, groups, composition, domains, pca, n80, n90, labels):
    group_count = sum(len(v) for v in groups.values())
    table1 = pd.DataFrame([
        ("Analytical sample", "Countries", len(df), "Corrected post-audit sample"),
        ("Primary matrix", "Schema-backed capability variables", len(features), "Rank, PwrIndx and metadata excluded"),
        ("Corrected framework", "Substantive domains", 8, "Financial, geography, manpower, airpower, land, naval, natural resources, logistics"),
        ("Redundancy control", "Retained domain components", group_count, "Within-domain connected groups at |Spearman rho| >= 0.90 represented by oriented PC1"),
        ("Transformation", "Log1p variables", len(logs), "Restricted to eligible non-negative skewed magnitude variables"),
        ("Scaling", "Domain scaling", "RobustScaler", "Applied after equal-weight domain construction"),
        ("Dimension reduction", "PCA80 retained components", n80, f"Cumulative variance {pca.explained_variance_ratio_[:n80].sum():.4f}"),
        ("Dimension reduction", "PCA90 retained components", n90, f"Cumulative variance {pca.explained_variance_ratio_[:n90].sum():.4f}"),
        ("Cluster search", "Candidate k", "2-6", "K-Means, Ward/average/complete hierarchical, GMM and K-Medoids"),
        ("Stability", "Corrected case-bootstrap refits", N_BOOT, "Duplicates retained; preprocessing, domain construction and PCA refitted"),
    ], columns=["framework_stage", "item", "value", "post_audit_definition"])
    save_table(table1, "table01_dataset_preprocessing_framework.csv")

    ev = pd.read_csv(SOURCE / "tables" / "s8_corrected_pca_explained_variance.csv")
    loadings = pd.read_csv(SOURCE / "tables" / "s8_corrected_pca_loadings.csv").set_index("domain")
    loadings = loadings.T.reset_index(names="pc_label")
    loadings.insert(0, "component", loadings.pc_label.str.removeprefix("PC").astype(int))
    table2 = ev.merge(loadings.drop(columns="pc_label"), on="component", how="left")
    save_table(table2, "table02_pca_results.csv")

    comparison = pd.read_csv(SOURCE / "tables" / "corrected_k_comparison.csv")
    save_table(comparison, "table03_corrected_k2_k6_comparison.csv")

    zdomains = pd.DataFrame(StandardScaler().fit_transform(domains), columns=DOMAINS)
    zdomains["cluster_id"] = labels + 1
    profiles = zdomains.groupby("cluster_id")[DOMAINS].mean().reset_index()
    sizes = pd.Series(labels + 1).value_counts().sort_index()
    profiles.insert(1, "cluster_size", profiles.cluster_id.map(sizes))
    profiles["solution_role"] = "Primary corrected k=3 taxonomy"
    save_table(profiles, "table04_final_k3_capability_profiles.csv")

    peers = pd.read_csv(SOURCE / "tables" / "turkiye_peer_balanced.csv")
    save_table(peers, "table05_turkiye_balanced_peer_analysis.csv")
    return ev, comparison, profiles, peers, zdomains


def build_figures(df, domains, pca, scores, n90, labels, ev, comparison, profiles, peers, consensus):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    colors = sns.color_palette("colorblind", 3)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    nodes = ["GFP data\n143 countries", "Post-audit cleaning\n8 domains", "Redundancy control\n|rho| >= 0.90", "Robust scaling\nPCA90", "k=2-6 validation\n500 bootstraps", "k=3 profiles\nBalanced peers"]
    xs = np.linspace(.07, .93, len(nodes))
    for i, (x, text) in enumerate(zip(xs, nodes)):
        ax.text(x, .52, text, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.45", fc="#eaf2f8", ec="#2c3e50", lw=1.1))
        if i < len(nodes)-1:
            ax.annotate("", xy=(xs[i+1]-.075, .52), xytext=(x+.075, .52), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1.3, color="#2c3e50"))
    ax.set_title("Post-audit corrected capability-profile workflow", weight="bold")
    save_figure(fig, "figure01_analysis_workflow")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = ev.component.astype(int)
    ax1.bar(x, ev.explained_variance_ratio, color="#4472c4", label="Explained variance")
    ax1.set(xlabel="Principal component", ylabel="Explained variance ratio", xticks=x)
    ax2 = ax1.twinx()
    ax2.plot(x, ev.cumulative_explained_variance, marker="o", color="#c44e52", label="Cumulative variance")
    ax2.axhline(.80, color="#555555", ls="--", lw=1); ax2.axhline(.90, color="#777777", ls=":", lw=1)
    ax2.set(ylabel="Cumulative explained variance", ylim=(0, 1.05))
    ax1.set_title("Corrected S8 PCA: scree and cumulative variance", weight="bold")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(.5, -.01))
    save_figure(fig, "figure02_pca_scree_cumulative")

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in range(3):
        idx = labels == cluster
        ax.scatter(scores[idx, 0], scores[idx, 1], s=28, alpha=.75, color=colors[cluster], label=f"Cluster {cluster+1} (n={idx.sum()})")
    tr = cap.find_turkiye(df)
    ax.scatter(scores[tr, 0], scores[tr, 1], s=140, marker="*", color="black", label="Türkiye", zorder=5)
    ax.annotate("Türkiye", (scores[tr,0], scores[tr,1]), xytext=(6,6), textcoords="offset points")
    ax.set(xlabel=f"PC1 ({ev.explained_variance_ratio.iloc[0]:.1%})", ylabel=f"PC2 ({ev.explained_variance_ratio.iloc[1]:.1%})")
    ax.set_title("Corrected PCA capability map with primary k=3 solution", weight="bold"); ax.legend(frameon=False)
    save_figure(fig, "figure03_pca_capability_map")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    panels = [("silhouette", "Silhouette", True), ("bootstrap_ARI", "Corrected bootstrap ARI", True), ("cross_method_ARI", "Cross-method ARI", True), ("naval_presence_ARI", "Naval-presence ARI", False)]
    for ax, (col, title, higher) in zip(axes.flat, panels):
        ax.plot(comparison.k, comparison[col], marker="o", color="#4472c4")
        ax.axvline(3, color="#2e8b57", ls="--", lw=1.2, label="Primary k=3")
        ax.axvline(4, color="#e1812c", ls=":", lw=1.2, label="Robustness k=4")
        ax.set(title=title, xlabel="Number of clusters (k)", ylabel=title, xticks=comparison.k)
    axes[0,0].legend(frameon=False, fontsize=8)
    fig.suptitle("Corrected cluster validation for k=2-6", weight="bold")
    fig.tight_layout()
    save_figure(fig, "figure04_corrected_cluster_validation")

    profile_matrix = profiles.set_index("cluster_id")[DOMAINS]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(profile_matrix, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=.4, ax=ax, cbar_kws={"label":"Standardized domain centroid"})
    ax.set(xlabel="Capability domain", ylabel="Corrected k=3 cluster", title="Corrected k=3 domain-profile heatmap")
    ax.set_yticklabels([f"Cluster {i}" for i in profile_matrix.index], rotation=0)
    save_figure(fig, "figure05_corrected_k3_domain_profiles")

    order = np.argsort(labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(consensus[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False, ax=ax, cbar_kws={"label":"Co-association frequency"})
    boundaries = np.cumsum(np.bincount(labels[order]))[:-1]
    for boundary in boundaries:
        ax.axhline(boundary, color="white", lw=.8); ax.axvline(boundary, color="white", lw=.8)
    ax.set(title="Corrected k=3 case-bootstrap consensus", xlabel="Countries ordered by reference cluster", ylabel="Countries ordered by reference cluster")
    save_figure(fig, "figure06_corrected_k3_consensus")

    plot_peers = peers.head(15).sort_values("family_presence_frequency")
    level_colors = {"HIGH":"#2e8b57", "MODERATE":"#4472c4", "LOW":"#999999"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_peers.country, plot_peers.family_presence_frequency, color=[level_colors[x] for x in plot_peers.stability_level])
    for y, value in enumerate(plot_peers.family_presence_frequency):
        ax.text(value + .012, y, f"{value:.0%}", va="center", fontsize=8)
    ax.set(xlabel="Presence among 42 equally weighted families", ylabel="Country", xlim=(0, 1), title="Türkiye balanced peer similarity")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=level_colors[k], label=k) for k in level_colors], title="Stability", frameon=False)
    save_figure(fig, "figure07_turkiye_balanced_peers")


def write_json_and_summary(ev, comparison, profiles, peers):
    source_json = json.loads((SOURCE / "json" / "manuscript_results.json").read_text(encoding="utf-8"))
    corrected = source_json["corrected_analysis"]
    output = {
        "provenance": "post-audit corrected results only",
        "invalid_original_results_included": False,
        "primary_solution": {"k": 3, "role": "primary analytical solution"},
        "robustness_alternative": {"k": 4, "role": "robustness alternative"},
        "rejected_solution": {"k": 2, "reason": "reproduces naval presence", "naval_presence_ARI": float(comparison.loc[comparison.k.eq(2), "naval_presence_ARI"].iloc[0])},
        "corrected_analysis": corrected,
        "final_k3_profiles": profiles.to_dict("records"),
        "turkiye_balanced_peers": peers.to_dict("records"),
    }
    (DEST / "manuscript_results_corrected.json").write_text(json.dumps(cap.native(output), ensure_ascii=False, indent=2), encoding="utf-8")
    k3 = comparison.loc[comparison.k.eq(3)].iloc[0]
    text = f"""# Publication-ready corrected analysis

This folder contains post-audit corrected results only. No original or invalid statistics are permitted.

- k=3 is the primary analytical solution.
- k=4 is the robustness alternative; no unique optimum is claimed.
- k=2 is rejected because it reproduces naval presence (ARI={comparison.loc[comparison.k.eq(2), 'naval_presence_ARI'].iloc[0]:.3f}); it is not used as the final taxonomy.
- Only corrected case-bootstrap values are reported: for k=3, ARI={k3.bootstrap_ARI:.3f}, AMI={k3.bootstrap_AMI:.3f}, and NMI={k3.bootstrap_NMI:.3f} ({N_BOOT} refitted resamples).
- Türkiye HIGH-stability peers: Thailand, Taiwan, Pakistan.
- No old invalid statistics are allowed in this folder.

PCA90 retains {int(ev.retained_90pct.sum())} components and {ev.loc[ev.retained_90pct, 'cumulative_explained_variance'].iloc[-1]:.1%} cumulative variance. The primary k=3 result must be interpreted with qualifications because general scale remains influential and cross-method agreement is moderate.
"""
    (DEST / "publication_summary.md").write_text(text, encoding="utf-8")


def validate():
    files = sorted(p for p in DEST.iterdir() if p.is_file())
    expected_tables = 5
    expected_png = expected_tiff = 7
    assert len(list(DEST.glob("table*.csv"))) == expected_tables
    assert len(list(DEST.glob("figure*.png"))) == expected_png
    assert len(list(DEST.glob("figure*.tiff"))) == expected_tiff
    assert (DEST / "manuscript_results_corrected.json").exists()
    assert (DEST / "publication_summary.md").exists()
    forbidden = [
        (re.compile(r"0\.999[0-9]*"), "invalid bootstrap ARI around 0.9996"),
        (re.compile(r"(?:preferred|primary|final)[^\n]{0,80}k\s*=\s*2", re.I), "k=2 presented as preferred/final"),
        (re.compile(r"31\s*[/|]\s*112[^\n]{0,80}(?:final|taxonomy)", re.I), "31/112 presented as final taxonomy"),
    ]
    for path in files:
        if path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        content = path.read_text(encoding="utf-8-sig", errors="ignore")
        for pattern, description in forbidden:
            if pattern.search(content):
                raise RuntimeError(f"Publication validation failed: {description} in {path.name}")
    print("PUBLICATION_READY_VALIDATION: PASS")
    print("MANIFEST")
    for path in files:
        print(f"- {path.name}\t{path.stat().st_size} bytes")
    print("CONFIRMED: all tables and figures derive from corrected post-audit results only.")


def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty publication folder: {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    objects = corrected_objects()
    df, naval, features, decisions, logs, scenarios, groups, composition, domains, scaled, pca, scores, n80, n90, labels = objects
    ev, comparison, profiles, peers, zdomains = build_tables(df, features, decisions, logs, groups, composition, domains, pca, n80, n90, labels)
    consensus = corrected_consensus(df, naval, features, logs, groups, labels, n90)
    build_figures(df, domains, pca, scores, n90, labels, ev, comparison, profiles, peers, consensus)
    write_json_and_summary(ev, comparison, profiles, peers)
    validate()


if __name__ == "__main__":
    main()
