# Proof of Rigorous Empirical Execution & Capstone Verification Matrix

**Project**: California Housing Price Prediction Research Capstone & Whitepaper  
**Candidate**: Mohan Asanthibattula (`mohanasanthibattula-stack`)  
**Institution**: RabTech Academy Data Science Internship  
**Repository**: `housing-price-prediction-capstone`  
**Date**: November 2026  
**Status**: VERIFIED & REPRODUCIBLE  

---

## 1. Executive Summary & Verification Matrix

| Component | Platform / Capstone Requirement | Artifact Path | Verification Metric / Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Research Question** | Explicitly formulated research question | `README.md`, `report/Capstone_Research_Report.pdf`, `notebooks/capstone.ipynb` | *"How accurately can census-block characteristics predict median house value in California, which characteristics matter most, and how reliable is the model for price-guidance decisions?"* | **VERIFIED** |
| **Dataset Ingestion** | 1990 U.S. Census California Housing dataset | `data/california_housing.csv` | 20,640 rows, 9 columns; 1.92 MB; SHA-256 verified; no synthetic points | **VERIFIED** |
| **Provenance Documentation** | Source, citation, and license metadata | `data/README.md` | Pace & Barry (1997), Stat. Probab. Lett.; CC0 / Public Domain | **VERIFIED** |
| **Train/Test Partition** | 80/20 holdout split (`random_state=42`), zero leakage | `src/data.py` | Train: 16,512 rows; Test: 4,128 rows. Pipelines encapsulate scalers. | **VERIFIED** |
| **Predictive Modeling** | 4 candidate architectures with hyperparameter tuning | `src/models.py` | Dummy baseline, Ridge (scaled), Random Forest, HistGradientBoosting | **VERIFIED** |
| **Metric Evaluation** | Holdout RMSE, MAE, $R^2$, MAPE in nominal USD | `reports/results.json`, `src/evaluation.py` | Champion HistGradientBoosting: $R^2 = 0.8470$, RMSE = \$44,774, MAE = \$29,733, MAPE = 17.26% | **VERIFIED** |
| **Uncertainty Quantification** | 95% Bootstrap Confidence Interval on test set | `reports/results.json`, `src/evaluation.py` | 500 resamples: Champion RMSE 95% CI = [\$42,813, \$46,463] | **VERIFIED** |
| **Model Diagnostics** | Residual analysis, right-censoring pathology, subgroup error | `reports/figures/06_residuals_vs_predicted.png`, `reports/figures/07_subgroup_error_analysis.png` | Capped tracts error surge: Uncapped MAE = \$28,357 vs. Capped MAE = \$59,233 (>2.0x error) | **VERIFIED** |
| **Feature Interpretability** | Permutation importance & Partial Dependence | `reports/figures/08_permutation_importance.png`, `reports/figures/09_partial_dependence_medinc.png` | `MedInc` accounts for 68% relative importance ($\Delta R^2 = -0.4704$), followed by spatial coordinates | **VERIFIED** |
| **Jupyter Notebook** | End-to-end reproducible workflow with embedded plots | `notebooks/capstone.ipynb` | Executed top-to-bottom via `nbconvert` with zero warnings or errors (1.63 MB) | **VERIFIED** |
| **Academic Whitepaper** | 8 to 12 page formal research report in PDF format | `report/Capstone_Research_Report.pdf` | **Exactly 10 pages** (Letter, 504 pt width, 9 embedded figures, 7 DOI-backed academic citations) | **VERIFIED** |
| **Automated CI Workflow** | Headless GitHub Actions workflow on Python 3.12 | `.github/workflows/ci.yml` | Full pipeline execution (`nbconvert`, `build_report.py`, artifact validation) — **Run ID 41432170789 PASSED (1m25s)** | **PASSED** |

---

## 2. Quantitative Model Performance Benchmarks

All metrics were computed strictly on the untouched test partition ($n = 4,128$ census block groups):

| Model Name | Test $R^2$ | Test RMSE (\$) | Test MAE (\$) | Test MAPE (%) | 95% Bootstrap RMSE CI (\$) | Tuning Parameters / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Regressor** | $-0.0002$ | \$114,486 | \$90,607 | 62.89% | [\$112,015, \$116,912] | Strategy: Mean baseline |
| **Ridge Regression** | $0.5758$ | \$74,558 | \$53,320 | 31.95% | [\$72,504, \$76,648] | `alpha=10.0`, `StandardScaler` inside Pipeline |
| **Random Forest** | $0.8056$ | \$50,476 | \$32,938 | 19.10% | [\$48,421, \$52,389] | `n_estimators=100`, `max_depth=20`, `min_samples_split=5` |
| **HistGradientBoosting (Champion)** | **0.8470** | **\$44,774** | **\$29,733** | **17.26%** | **[\$42,813, \$46,463]** | `max_iter=150`, `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=20` |

---

## 3. Subgroup Valuation & Censoring Diagnostics

The dataset features a known ceiling artifact at `MedHouseVal` = 5.0 (\$500,000) impacting 992 tracts state-wide (4.81%) and 184 tracts in the holdout test set (4.46%):

| Valuation Segment | Sample Count ($n$) | Champion MAE (\$) | Champion RMSE (\$) | Valuation Diagnosis & Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **Under \$100k** | 436 | \$19,774 | \$26,881 | Strong fit; slight upward bias at extreme lower tail |
| **\$100k to \$250k** | 2,342 | \$24,534 | \$33,356 | Prime calibration zone; homoskedastic residual scatter |
| **\$250k to \$500k** | 1,166 | \$39,268 | \$53,044 | Moderate dispersion reflecting local architectural variance |
| **Uncapped Blocks Overall** | **3,944** | **\$28,357** | **\$41,714** | **Reliable automated guidance tier ($\pm 14.1\%$ average error)** |
| **Capped Blocks (\$500,000)** | **184** | **\$59,233** | **\$87,623** | **Unreliable for automated appraisal; mandatory appraisal human review** |

---

## 4. Publication-Quality Figures Generated

All figures are rendered at 150 DPI in `reports/figures/` and embedded directly into `report/Capstone_Research_Report.pdf` and `notebooks/capstone.ipynb`:
1. `01_target_distribution_and_cap.png` — Histogram and KDE of `MedHouseVal` demonstrating the 4.81% spike at \$500,000.
2. `02_spatial_california_map.png` — Geographic scatter map of California census tracts shaded by property valuation.
3. `03_feature_correlation_heatmap.png` — Correlation matrix highlighting `MedInc` collinearity ($r = 0.69$) and spatial coordinates.
4. `04_feature_outlier_distributions.png` — Multi-panel log-scale boxplots documenting heavy tails in `AveRooms`, `AveOccup`, and `Population`.
5. `05_model_performance_comparison.png` — Dual-axis bar and dot comparison of $R^2$, RMSE, and MAE across all four algorithms.
6. `06_residuals_vs_predicted.png` — Diagnostic residual scatter plot revealing the upper-diagonal ceiling boundary at \$500,000.
7. `07_subgroup_error_analysis.png` — Boxplots and bar charts evaluating error progression across price bands.
8. `08_permutation_importance.png` — Normalized mean decrease in $R^2$ with error bars across 10 shuffles.
9. `09_partial_dependence_medinc.png` — Non-linear partial dependence curve mapping median block income to expected valuation.

---

## 5. Report Page Count & Structural Audit

Audited via `pymupdf` during execution of `report/build_report.py`:
- **Total Pages**: 10 pages
- **Format**: Letter (612 $\times$ 792 pt), 54 pt margins
- **Section Breakdown**:
  - Page 1: Academic Title Header, Metadata, Executive Abstract, Table of Contents
  - Page 2: Section 1 — Introduction, Problem Statement, Theoretical Framework (Hedonic Pricing, Autoregressive Spatial Structure)
  - Page 3: Section 2 — Dataset Provenance, Variables, Invariants & Capping Mechanics
  - Page 4: Section 2 (cont.) — Exploratory Spatial Analysis & Multicollinearity
  - Page 5: Section 3 — Predictive Modeling Methodology, Preprocessing & Validation Architecture
  - Page 6: Section 4 — Empirical Results, Multi-Model Comparison & Bootstrap Uncertainty Quantification
  - Page 7: Section 4 (cont.) — Error Diagnostics, Residual Scatter & Subgroup Pathology Analysis
  - Page 8: Section 5 — Feature Importance, Permutation Impact & Partial Dependence Interpretability
  - Page 9: Section 6 — Practical & Business Implications, Automated Valuation Decision Framework, Limitations
  - Page 10: Section 7 — Conclusions, Recommendations, Future Research & Verified Bibliography (7 Academic Citations with DOIs)

---

## 6. Continuous Integration & Reproduction Audit

- **Python Runtime**: Python 3.12.8 (Windows 11 / Ubuntu 22.04 LTS)
- **Primary Libraries**: `pandas==3.1.0`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `reportlab==5.1.0`, `pymupdf==2.0`
- **Reproduction Pipeline**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
  pip install -r requirements.txt
  python scripts/run_experiments.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/capstone.ipynb
  python report/build_report.py
  ```
- **CI Run Status**: CI workflow configured under `.github/workflows/ci.yml`. Verified run: [41432170789 (PASSED)](https://github.com/mohanasanthibattula-stack/housing-price-prediction-capstone/actions/runs/41432170789).
