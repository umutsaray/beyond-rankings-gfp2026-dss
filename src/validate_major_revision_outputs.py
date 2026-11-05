from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path("outputs/capability_profiles/major_revision")
REQUIRED = [
    "major_revision_results.json",
    "major_revision_report.md",
    "task1_excluded_countries.csv",
    "task1_variable_dictionary_76.csv",
    "task2_eta_squared_reconstruction.csv",
    "task2_alternative_cluster_comparison_k2_k6.csv",
    "task2_domain_variance_contributions.csv",
    "task2_PC1_loading_robust_vs_unit_variance.csv",
    "task2_k2_landlocked_overlap.json",
    "task3_neighbor_set_comparison.csv",
    "taskA_k3_clusterwise_jaccard_summary.csv",
    "taskA_k3_six_method_pairwise_agreement_15pairs.csv",
    "taskB_label_matching_audit.json",
    "taskC_peer_topN_sensitivity.csv",
    "taskD_pca_reproducibility_audit.json",
    "taskE_silhouette_discrepancy_audit.json",
    "taskF_reference_method_fit_audit.csv",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    assert not missing, f"Missing outputs: {missing}"
    dictionary = pd.read_csv(ROOT / "task1_variable_dictionary_76.csv")
    assert len(dictionary) == 76 and dictionary.variable.nunique() == 76
    excluded = pd.read_csv(ROOT / "task1_excluded_countries.csv")
    assert set(excluded.country_official) == {"Lithuania", "North Macedonia"}
    alternatives = pd.read_csv(ROOT / "task2_alternative_cluster_comparison_k2_k6.csv")
    assert set(alternatives.k) == {2, 3, 4, 5, 6} and alternatives.representation.nunique() == 2
    pairs = pd.read_csv(ROOT / "taskA_k3_six_method_pairwise_agreement_15pairs.csv")
    assert len(pairs) == 15
    peer = pd.read_csv(ROOT / "taskC_peer_topN_sensitivity.csv")
    assert set(peer.top_n) == {5, 10, 15}
    result = json.loads((ROOT / "major_revision_results.json").read_text(encoding="utf-8"))
    assert abs(result["task2"]["eta_squared"] - 0.6548217399327382) < 1e-12
    assert abs(result["task2"]["naval_variance_share_robust_PCA_input"] - 0.439244172701782) < 1e-12
    assert result["task2"]["k2_overlap"]["same_31_countries"] is True
    rows = []
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.name != "manifest.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(ROOT / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"PASS: {len(rows)} files validated; manifest written")


if __name__ == "__main__":
    main()
