# Beyond Rankings — Major Revision Analysis Report

## Scope and reproducibility

All new results were produced by `src/run_major_revision_analyses.py`, which imports the existing project pipeline rather than redesigning it. The analysis uses deterministic seeds (`RANDOM_STATE=42` for full-sample clustering and the existing corrected bootstrap seed schedule) and writes only to `outputs/capability_profiles/major_revision/`. The source dataset and prior analysis outputs were not overwritten.

The official comparison source is the GFP 2026 ranking page: https://www.globalfirepower.com/countries-listing.php (accessed 5 November 2026).

## Task 1 — 143 versus 145 countries and the 76-variable dictionary

**Code:** `src/run_major_revision_analyses.py` (`country_reconciliation`, `data_dictionary`).

**Numerical result:** GFP officially lists 145 countries, whereas the local raw PDF corpus, raw extraction, cleaned dataset, and analysis matrix contain 143. Lithuania (official rank 88) and North Macedonia (official rank 114) have no PDF in `gfp_pdfs` and no row in the raw extraction. They were therefore absent at source acquisition/raw extraction; neither was removed by missingness screening or modeling. Benin and Kosovo are present, but their extracted ranks are shifted by one position (137 rather than 138 and 138 rather than 139, respectively), which is a separate rank-parsing issue. The analytical matrix contains exactly 76 included variables.

**Outputs:** `task1_excluded_countries.csv`, `task1_rank_extraction_issues.csv`, and `task1_variable_dictionary_76.csv`.

**Recommended manuscript text:** “The official GFP 2026 ranking comprised 145 countries. The locally acquired country-PDF corpus contained 143 records because Lithuania and North Macedonia were absent at the acquisition stage; no country was subsequently removed by feature-missingness screening or clustering.” Add the 76-row dictionary as a supplementary data table.

## Task 2 — Direct tests of scale and naval dominance

### 2.1 Exact eta-squared calculation

**Code:** `scale_index`, `eta_squared`, and the reconstruction block in `src/run_major_revision_analyses.py`.

**Numerical result:** The scale variable is not PC1 of the eight-domain PCA. It is a separate general-scale index: PC1 of 72 nonnegative magnitude indicators after `log1p` and `StandardScaler`, oriented so the sum of its loadings is positive. For the reference k=3 labels, eta-squared is computed as

`eta² = SS_between / SS_total = sum_c[n_c(mean_c - grand_mean)²] / sum_i[(x_i - grand_mean)²] = 0.65482174`.

**Output:** `task2_eta_squared_reconstruction.csv` and `task2_general_scale_index_features.csv`.

**Recommended manuscript text:** “Scale association was quantified as eta² between the k-means partition and an independently constructed 72-indicator general-scale PC1; eta²=0.655 was obtained as between-cluster sum of squares divided by total sum of squares.”

### 2.2 Naval-deletion and unit-variance alternatives

**Code:** `fit_representation`, `cluster_grid`, and `peer_top`.

**Numerical result:** Removing the naval domain changes k=3 from 31/55/57 to 41/45/57, with ARI=0.3335, AMI=0.4158, and NMI=0.4235 versus the original labels; silhouette falls from 0.4609 to 0.2801. Under this seven-domain representation, Thailand and Pakistan appear in the top 10 under all three distance metrics, whereas Taiwan appears under none.

Unit-variance scaling retains a much closer k=3 partition (31/52/60; ARI=0.9292, AMI=0.9216, NMI=0.9226), although silhouette is 0.3561. The naval PC1 eigenvector coefficient falls from 0.7457 under RobustScaler to 0.2267 after unit-variance scaling. Thailand and Pakistan remain top-10 neighbors under all three distances; Taiwan does not enter the top 10 under any of the three.

**Outputs:** `task2_alternative_cluster_comparison_k2_k6.csv`, `task2_alternative_turkiye_neighbors_top10.csv`, `task2_key_peer_retention.csv`, and `task2_PC1_loading_robust_vs_unit_variance.csv`.

**Recommended manuscript text:** “The k=3 taxonomy was highly stable to unit-variance domain scaling (ARI=0.929) but only moderately preserved after complete naval-domain deletion (ARI=0.333). Thus, the tripartite structure is not solely a scaling artifact, yet its detailed membership depends materially on maritime information.” Do not claim that all three headline peers are naval-deletion robust; Taiwan is not.

### 2.3 Exact naval share of PCA-input variance

**Code:** covariance-trace decomposition in `src/run_major_revision_analyses.py`.

**Numerical result:** In the RobustScaler-transformed eight-domain matrix, naval variance is 43.9244% of the total PCA-input variance. This is calculated directly as the naval diagonal element of the sample covariance matrix divided by its trace; equivalently, each diagonal variance can be reconstructed by summing `eigenvalue_j × eigenvector_coefficient_j²` across all PCs. After unit-variance scaling, the share is exactly 12.5%.

**Outputs:** `task2_domain_variance_contributions.csv` and `task2_unit_variance_contributions.csv`.

**Recommended manuscript text:** “The naval domain contributed 43.9% of the total covariance-trace variance entering the primary PCA, compared with 12.5% under unit-variance normalization.”

### 2.4 RobustScaler and outliers

**Code:** `mahalanobis_outlier_comparison`.

**Numerical result:** Using the chi-square 99% cutoff for eight dimensions (20.090), China, Iceland, and the United States are flagged both before and after RobustScaler. Their Mahalanobis distances are unchanged up to floating-point precision because an invertible affine rescaling does not remove observations or eliminate multivariate leverage.

**Outputs:** `task2_mahalanobis_before_after_robust_scaler.csv` and `task2_mahalanobis_before_after_summary.csv`.

**Recommended manuscript text:** “RobustScaler reduced marginal scale sensitivity but did not remove multivariate leverage: the same three observations exceeded the 99% Mahalanobis cutoff before and after scaling.” Replace any statement implying automatic outlier protection.

### 2.5 Meaning of the k=2 partition

**Code:** k=2/k=3 overlap block in `src/run_major_revision_analyses.py`.

**Numerical result:** The 31-country k=2 group is exactly the same set as the 31-country k=3 cluster (Jaccard=1.000). k=2 also matches binary naval presence exactly (ARI=1.000), while overall k=2-versus-k=3 ARI is 0.4362.

**Output:** `task2_k2_landlocked_overlap.json`.

**Recommended manuscript text:** Reframe k=2 as a real but substantively insufficient maritime-access split, not as a purely computational artifact: “The binary solution recovered a genuine structural landlocked/maritime distinction, but it was rejected as the final taxonomy because it failed to differentiate the 112 maritime-present cases.”

## Task 3 — Added value beyond GFP rank

**Code:** `ranking_and_baselines`.

**Numerical result:** Türkiye’s GFP rank-neighbor top 10 are United Kingdom, Italy, Japan, Brazil, France, Germany, South Korea, Indonesia, India, and Pakistan. The balanced profile top 10 are Thailand, Taiwan, Pakistan, Italy, France, Egypt, Germany, Japan, South Korea, and Spain. Six countries overlap. Brazil, India, Indonesia, and the United Kingdom are rank-only; Thailand, Taiwan, Egypt, and Spain are profile-only.

The simple PCA-free Euclidean baseline on the raw eight corrected domain scores returns Egypt, Pakistan, Iran, Ukraine, Saudi Arabia, Algeria, South Korea, Thailand, Italy, and Japan. Six overlap with the full pipeline. Taiwan, France, Germany, and Spain occur only in the full pipeline; Iran, Ukraine, Saudi Arabia, and Algeria occur only in the simple baseline.

**Outputs:** `task3_gfp_rank_neighbors_top10.csv`, `task3_full_pipeline_profile_neighbors_top10.csv`, `task3_simple_8domain_euclidean_top10.csv`, `task3_neighbor_set_comparison.csv`, and `task3_nonoverlap_domain_differences.csv`.

**Recommended manuscript text:** “Only six of Türkiye’s ten rank-adjacent countries were recovered by the stability-balanced profile method. The method additionally identified Thailand, Taiwan, Egypt, and Spain, demonstrating that ordinal proximity and multivariate structural similarity are not interchangeable.” Use the domain-difference table to explain each non-overlap; for example, Taiwan differs most in geography, naval capability, and logistics, whereas the United Kingdom differs most in land, manpower, and logistics.

## Task A — Expanded bootstrap reporting

**Code:** corrected 500-iteration `bootstrap_corrected`, followed by cluster- and country-level summaries.

**Numerical result:** Mean cluster-wise Jaccard scores are 0.844 (57-country cluster), 0.860 (31-country cluster), and 0.903 (55-country cluster). OOB ARI mean=0.8215, SD=0.2197, median=0.9303, empirical 95% interval=[0.2845, 1.000]. The least stable countries are Portugal (0.420), Democratic Republic of the Congo (0.526), Sudan (0.586), Netherlands (0.646), and Sweden (0.662). The 15 pairwise method ARIs range from 0.051 (average-versus-complete linkage) to 0.845 (Ward-versus-GMM).

**Outputs:** `taskA_k3_bootstrap_all_iterations.csv`, `taskA_k3_clusterwise_jaccard_500.csv`, `taskA_k3_clusterwise_jaccard_summary.csv`, `taskA_k3_country_membership_stability.csv`, `taskA_k3_most_unstable_countries_top20.csv`, `taskA_k3_out_of_bag_ARI_summary.csv`, and `taskA_k3_six_method_pairwise_agreement_15pairs.csv`.

**Recommended manuscript change:** Add a supplementary bootstrap table containing cluster-wise Jaccard, OOB ARI uncertainty, the ten least stable countries, and the full 6×6 agreement matrix or its 15 unique pairs.

## Task B — Label matching for Türkiye stability

**Code:** `audit.align_labels`.

**Numerical result:** Each bootstrap label vector is matched to the full-sample reference through the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) applied to the negative contingency matrix. Türkiye then remains in its reference-matched cluster in 500/500 iterations. The 100% value is therefore permutation-adjusted, not based on raw cluster identifiers.

**Output:** `taskB_label_matching_audit.json`.

**Recommended manuscript text:** “Bootstrap labels were aligned to the full-sample solution by Hungarian maximum-overlap assignment before country-specific membership recurrence was computed.”

## Task C — S8-corrected versus the other six peer scenarios

**Code:** `peer_scenario_sensitivity`.

**Numerical result:** For top-10 selection, the S8-corrected family alone (2 spaces × 3 distances) identifies Egypt, Ukraine, Thailand, and Iraq in all six configurations. Thailand appears in 6/6, but Taiwan and Pakistan appear in only 1/6 each. Across the other six scenarios, Thailand appears in 31/36, Taiwan in 30/36, and Pakistan in 28/36. Thus, the published 42-family headline for Taiwan and Pakistan is driven mainly by the non-S8 scenarios. Sensitivity for top 5/10/15 is reported separately. Mean rank is conditional on a country entering the selected top-N list; absent configurations do not contribute to the mean.

**Outputs:** `taskC_peer_all_configurations_top15.csv` and `taskC_peer_topN_sensitivity.csv`.

**Recommended manuscript change:** Separate the S8-corrected peer result from the pooled 42-configuration robustness ensemble. Describe Thailand as robust in both; describe Taiwan and Pakistan as ensemble-stable but not S8-specific top-10 peers.

## Task D — PCA reproducibility details

**Code:** audited `connected_groups`, `orient_pc1`, `fit_corrected_domains`, and `bootstrap_corrected`.

**Numerical result:** Rank and PwrIndx variables are excluded rather than reversed. No blanket reverse-coding is applied; negative deficit indicators retain their source sign. Within-group PC1 is multiplied by -1 only when the sum of coefficients is negative. Correlation groups are graph-connected components, so chained links merge transitively. The full-sample PCA90 selects four components; bootstrap PCA is refitted at four components in every iteration but the component count is not reselected. Table 2 contains eigenvector coefficients, not variable–component correlations. Squared coefficients sum to 1 for every PC, with maximum orthonormality error 1.11×10^-15.

**Output:** `taskD_pca_reproducibility_audit.json`.

**Recommended manuscript change:** Add these implementation rules verbatim to Methods and label Table 2 coefficients as “PCA eigenvector coefficients.”

## Task E — Silhouette discrepancy

**Code:** exact recomputation from the corrected representation plus comparison of the two stored canonical tables.

**Numerical result:** Both `spearman_threshold_sensitivity.csv` and `corrected_k_comparison.csv` contain 0.460918799 for k=3. The manuscript value 0.4602 is a stale transcription error; there is no seed, subset, or preprocessing difference.

**Output:** `taskE_silhouette_discrepancy_audit.json`.

**Recommended manuscript change:** Use 0.4609 consistently in all tables and prose.

## Task F — Methodological fit of references

**Code/source audit:** manuscript reference list and the claims supported by each citation were compared with the publications’ stated scope.

**Result:** MacKinnon and Webb (2017) concerns wild cluster bootstrap inference for regression errors and is not an appropriate source for unsupervised cluster-discovery stability. Replace it for that claim with Hennig (2007), “Cluster-wise assessment of cluster stability,” doi:10.1016/j.csda.2006.11.025. Kim, Lee, and Kang (2018) is appropriate for multi-index internal cluster validation but not bootstrap stability. Hardin and Rocke (2005) is appropriate for robust Mahalanobis-distance diagnostics; retain it without claiming that RobustScaler removes outliers.

**Output:** `taskF_reference_method_fit_audit.csv`.

**Recommended manuscript change:** Replace the MacKinnon–Webb citation in the resampling-stability paragraph; narrow the Kim et al. citation to internal-validity triangulation; retain Hardin–Rocke for robust-distance diagnostics.

## Overall interpretation for the revision

The central k=3 claim survives the strongest equal-variance check (ARI=0.929) but is not invariant to deleting the naval domain (ARI=0.333). The revision should therefore present k=3 as a reproducible primary specification whose detailed membership remains substantively maritime-sensitive. The strongest “beyond rankings” evidence is the 6/10 overlap between rank and profile neighbors. The peer section must distinguish pooled ensemble stability from S8-corrected specificity, especially for Taiwan and Pakistan.
