# Beyond Rankings: GFP 2026 Decision-Support Pipeline

This repository contains the reproducible analysis code and derived Major Revision results for a Global Firepower (GFP) 2026 study of systemic military-capability profiles. The pipeline constructs eight substantive capability domains, applies robust preprocessing and PCA, compares multiple clustering algorithms across candidate values of *k*, evaluates case-bootstrap and cross-method stability, and performs a 42-configuration Türkiye peer analysis designed to identify structural similarity beyond ordinal GFP rank proximity.

## Repository contents

```text
.
├── src/                                           # Extraction, cleaning, PCA, clustering, audits, and revision analyses
├── outputs/
│   └── capability_profiles/
│       └── major_revision/                        # Committed derived CSV/JSON results and the revision report
├── requirements.txt
├── LICENSE                                        # MIT license for repository code
└── .gitignore
```

Key entry points:

- `src/create_capability_profile_analysis.py`: main preprocessing, PCA, clustering, validation, and peer-analysis pipeline.
- `src/audit_capability_profile_analysis.py`: methodological audit, including corrected resampling diagnostics.
- `src/correct_capability_profile_analysis.py`: post-audit corrected eight-domain analysis.
- `src/run_major_revision_analyses.py`: reviewer-requested naval deletion, unit-variance, outlier, ranking-baseline, bootstrap, peer-sensitivity, and reproducibility analyses.
- `src/validate_major_revision_outputs.py`: structural and numerical validation of the committed Major Revision results.
- `src/create_table_s0_extensions.py`: produces the reviewer-requested 76-to-56 component map and source-field definition supplement.
- `src/run_reviewer_tests_r24_r25_r26.py`: runs the reviewer-requested cluster-existence, method-bootstrap, stability-distribution, and peer-sensitivity tests.
- `src/run_remaining_reviewer2_tests.py`: runs the remaining R2.2, R2.6–R2.8, R2.11–R2.12, R2.16–R2.17 diagnostics and exports the complete k=3 membership table.
- `outputs/capability_profiles/major_revision/task1_variable_dictionary_76.csv`: the 76-variable data dictionary referenced as Table S0.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied `requirements.txt` is the project dependency specification. Some packages support analyses outside the core capability-profile workflow; the primary PCA/clustering pipeline relies principally on pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, PyMuPDF, and pdfplumber.

## Data source and redistribution policy

Source: [Global Firepower, 2026 Military Strength Ranking](https://www.globalfirepower.com/countries-listing.php), accessed 25 August 2026.

GlobalFirepower states that its published site content is unique to the website and “not for reuse in any form.” Its [site disclaimer](https://www.globalfirepower.com/disclaimer.php) also describes the material as general-reference content provided as-is. Accordingly:

- Raw GFP PDFs, downloaded HTML, and the direct country-level raw/cleaned GFP matrix are **not included** in this public repository.
- The MIT license applies only to the repository’s original code and documentation. It does not grant rights to GFP source content or third-party material.
- The repository includes the requested derived analytical outputs and the variable/schema dictionary, not a substitute redistribution of the GFP source dataset.
- To rerun extraction from scratch, users must obtain source material independently and ensure that their access, storage, and use comply with the source provider’s terms and applicable law.

Place locally obtained country PDFs in `gfp_pdfs/`. This directory is excluded by `.gitignore` and must not be committed.

## Reproduction sequence

### 1. Extract, clean, and validate locally supplied source files

```bash
python src/extract_gfp_pdfs.py --input-dir gfp_pdfs --output-dir outputs
python src/clean_gfp_dataset.py --output-dir outputs
python src/validate_gfp_dataset.py
```

The expected local clean-data path is:

```text
outputs/tables/gfp_2026_clean.csv
```

This file is intentionally ignored and is not distributed through the repository.

### 2. Run the main capability-profile pipeline

```bash
python src/create_capability_profile_analysis.py --mode final --output-dir outputs
```

### 3. Run the methodological audit and corrected specification

```bash
python src/audit_capability_profile_analysis.py
python src/correct_capability_profile_analysis.py
```

### 4. Run the Major Revision analyses

```bash
python src/run_major_revision_analyses.py
```

This writes reviewer-facing CSV/JSON results to:

```text
outputs/capability_profiles/major_revision/
```

### 5. Validate the revision outputs

```bash
python src/validate_major_revision_outputs.py
```

A successful run prints a `PASS` message and regenerates `manifest.csv` with file sizes and SHA-256 hashes.

## Randomness and reproducibility

- Main analysis seed: `RANDOM_STATE = 42`.
- Audit seed: `AUDIT_SEED = 240822`.
- Corrected-analysis seed: `SEED = 260822`.
- The corrected bootstrap uses 500 case-resampling iterations. For a candidate cluster count `k`, resampling starts from `SEED + k × 1000`; iteration-specific k-means seeds use `SEED + k × 1000 + iteration + 1`.
- Duplicate bootstrap draws are retained as case weights.
- Missing-value handling, redundancy-controlled domain construction, scaling, PCA, and k-means are refitted inside each corrected bootstrap iteration.
- Bootstrap labels are aligned to the full-sample reference partition through the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) before country-level recurrence is calculated.
- PCA90 selects four components in the full-sample corrected model. Bootstrap PCA is refitted at four components in every iteration; the component count is not reselected within each resample.

## Major Revision outputs

The committed results include:

- 145-versus-143 country reconciliation and extraction-stage exclusions.
- Table S0, the 76-variable data dictionary.
- The complete 76-to-56 within-domain correlation-group/component map and energy/readiness source-field definitions extending Table S0.
- Exact reconstruction of the general-scale eta-squared statistic.
- Seven-domain naval-deletion and eight-domain unit-variance alternatives for `k=2–6`.
- Exact naval share of PCA-input variance and PC1 loading comparisons.
- Mahalanobis diagnostics before and after RobustScaler.
- `k=2` landlocked-group overlap evidence.
- GFP-rank, full-pipeline, and simple eight-domain Türkiye neighbor comparisons.
- Cluster-wise Jaccard, out-of-bag ARI, country-level instability, and all 15 pairwise six-method agreements.
- S8-corrected versus other-scenario peer results and top-5/top-10/top-15 sensitivity.
- PCA implementation and silhouette-value audits.
- Methodological fit review for key references.
- Reviewer R2.4–R2.6 diagnostics: gap/null reference, dip tests, GMM BIC/ICL, method-specific 500-run bootstraps, silhouette comparison, bimodality and country-instability checks, and metric-specific Türkiye peer sensitivity.
- Remaining Reviewer 2 diagnostics: naval-label spot checks, null-overlap and PC1/rank relationships, wider redundancy thresholds, extreme-state leave-outs, Mahalanobis-space consistency, S5 and imputation sensitivity, complete membership, Türkiye k-sensitivity, source completeness, and Figure 7 peer-list reconciliation.

See [`outputs/capability_profiles/major_revision/major_revision_report.md`](outputs/capability_profiles/major_revision/major_revision_report.md) for the reviewer-oriented interpretation and manuscript recommendations.

## License

Original repository code and documentation are released under the [MIT License](LICENSE). GFP source data, site content, and any other third-party materials remain subject to their respective rights and terms and are not relicensed here.

