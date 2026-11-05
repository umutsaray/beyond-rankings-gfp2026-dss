from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path("outputs/capability_profiles/major_revision")
REQUIRED = [
    "major_revision_results.json",
    "major_revision_report.md",
    "task_variable_to_component_mapping.csv",
    "task_variable_definitions.csv",
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
    "reviewer_r24_gap_null_reference.csv",
    "reviewer_r24_pc1_pc2_dip_tests.csv",
    "reviewer_r24_gmm_bic_icl.csv",
    "reviewer_r24_method_bootstrap_500_iterations.csv",
    "reviewer_r24_method_bootstrap_summary.csv",
    "reviewer_r24_consensus_bootstrap_dependence.csv",
    "reviewer_r24_silhouette_k3_vs_k2_bootstrap_500.csv",
    "reviewer_r24_silhouette_k3_vs_k2_summary.csv",
    "reviewer_r24_method_landlocked31_and_remaining112.csv",
    "reviewer_r25_k3_bootstrap_distribution.json",
    "reviewer_r25_country_membership_stability.csv",
    "reviewer_r25_landlocked_stability_groups.csv",
    "reviewer_r25_landlocked_joint_switch_audit.json",
    "reviewer_r25_unstable_below080_cluster_composition.csv",
    "reviewer_r26_euclidean_cosine_peer_stability.csv",
    "reviewer_r26_s8_corrected_topn_sensitivity.csv",
    "reviewer_r26_baseline_overlap_and_egypt.json",
    "reviewer_tests_r24_r25_r26_summary.json",
    "reviewer_r22_naval_presence_spotcheck.csv",
    "reviewer_r22_coastline_naval_presence_disagreements.csv",
    "reviewer_r26_taiwan_alternative_recurrence.csv",
    "reviewer_r26_s8_domain_euclidean_distances.csv",
    "reviewer_r26_egypt_recurrence_check.json",
    "reviewer_r27_null_overlap_and_pc1_rank.csv",
    "reviewer_r27_k3_cluster_rank_distribution.csv",
    "reviewer_r28_spearman_threshold_070_095.csv",
    "reviewer_r211_extreme_leaveout_sensitivity.csv",
    "reviewer_r211_mahalanobis_domain_vs_pca.csv",
    "reviewer_r211_s5_winsorized_peer_overlap.json",
    "reviewer_r212_mean_imputation_sensitivity.json",
    "task_cluster_membership_k3.csv",
    "reviewer_r216_turkiye_k2_k6_stability.csv",
    "reviewer_r217_data_source_audit.csv",
    "reviewer_r217_figure7_peer_list_audit.csv",
    "reviewer_remaining_r2_summary.json",
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
    reviewer = json.loads((ROOT / "reviewer_tests_r24_r25_r26_summary.json").read_text(encoding="utf-8"))
    assert reviewer["specification"]["bootstrap_iterations"] == 500
    assert reviewer["r24"]["gap_1se_k"] == 3
    assert reviewer["r25"]["iterations_below_0_6"] == 97
    assert reviewer["r26"]["baseline"]["overlap_count"] == 7
    remaining = json.loads((ROOT / "reviewer_remaining_r2_summary.json").read_text(encoding="utf-8"))
    assert len(pd.read_csv(ROOT / "task_cluster_membership_k3.csv")) == 143
    assert remaining["r27"]["statistics"][1]["value"] == 6
    assert remaining["r212"]["changed_assignment_count"] == 0
    assert len(remaining["r217"]["moderate_plus"]) == 13
    rows = []
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.name != "manifest.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(ROOT / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"PASS: {len(rows)} files validated; manifest written")


if __name__ == "__main__":
    main()
