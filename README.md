# California Housing Price Prediction: Econometric Modeling & Machine Learning Capstone

[![CI Pipeline](https://github.com/mohanasanthibattula-stack/housing-price-prediction-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/mohanasanthibattula-stack/housing-price-prediction-capstone/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Whitepaper PDF](https://img.shields.io/badge/Research_Report-PDF_(10_Pages)-red.svg)](report/Capstone_Research_Report.pdf)

> **Research Question**: *"How accurately can census-block characteristics predict median house value in California, which characteristics matter most, and how reliable is the model for price-guidance decisions?"*

---

## Executive Summary & Key Results

This repository contains the complete end-to-end research capstone evaluating the predictive accuracy, structural feature drivers, and decision reliability of machine learning algorithms applied to aggregated census block group property valuations across California.

Evaluating four regression architectures on a strictly segregated 20% holdout test partition ($n = 4,128$) with zero data leakage:

| Model Architecture | Test $R^2$ | Test RMSE (\$) | Test MAE (\$) | Test MAPE (%) | 95% Bootstrap RMSE CI (\$) | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Baseline (Mean)** | $-0.0002$ | \$114,486 | \$90,607 | 62.89% | [\$112,015, \$116,912] | Naive benchmark |
| **Ridge Regression (Scaled)** | $0.5758$ | \$74,558 | \$53,320 | 31.95% | [\$72,504, \$76,648] | Linear parametric baseline |
| **Random Forest Regressor** | $0.8056$ | \$50,476 | \$32,938 | 19.10% | [\$48,421, \$52,389] | Non-linear ensemble |
| **HistGradientBoosting (Champion)** | **0.8470** | **\$44,774** | **\$29,733** | **17.26%** | **[\$42,813, \$46,463]** | **Production Champion** |

### Core Empirical Findings
1. **Predictive Capability**: The champion **Histogram-Based Gradient Boosting Regressor** accounts for **84.70% of variance** ($R^2 = 0.8470$) with a mean absolute error of **\$29,733** (17.26% MAPE) and an RMSE of **\$44,774** (95% Bootstrap CI: [\$42,813, \$46,463]).
2. **Key Feature Drivers**: Feature importance analysis reveals that **Median Block Income (`MedInc`) is the paramount explanatory factor**, contributing 68% of relative predictive power ($\Delta R^2 = -0.4704$), followed closely by spatial coordinates (`Longitude`: $\Delta R^2 = -0.1691$, `Latitude`: $\Delta R^2 = -0.1643$).
3. **Price-Guidance Reliability & Censoring Pathology**:
   - For **uncapped properties under \$500,000**, the model provides reliable automated price guidance with a low MAE of **\$28,357** (14.1% relative error).
   - For **properties subject to the \$500,000 census ceiling** (4.81% of the dataset), error surges by more than **208% to \$59,233 MAE** (\$87,623 RMSE). **Automated price decisions must not be executed autonomously on properties near or above this threshold without licensed human appraisal.**

---

## Formal Research Report (Whitepaper)

A formal **10-page academic research whitepaper** has been compiled and rendered in PDF format:
- **Direct PDF Link**: [`report/Capstone_Research_Report.pdf`](report/Capstone_Research_Report.pdf)
- **Preview Pages**: Available as high-resolution PNGs in [`report/page_previews/`](report/page_previews/)

### Report Structure:
1. **Title, Executive Abstract & Table of Contents** (Page 1)
2. **Introduction, Problem Statement & Theoretical Foundations** (Page 2)
   - Hedonic price modeling (Rosen, 1974) and spatial autoregressive dynamics (Pace & Barry, 1997).
3. **Dataset Provenance, Invariants & Exploratory Spatial Analysis** (Pages 3–4)
   - Analysis of 20,640 tracts, distribution skewness, heavy-tailed outliers, and California spatial value mapping.
4. **Predictive Modeling Methodology & Validation Architecture** (Page 5)
   - Strict 80/20 train/test partition, pipeline encapsulation to avoid data leakage, randomized CV hyperparameter tuning.
5. **Empirical Results & Multi-Model Comparison** (Page 6)
   - Evaluation metrics, statistical performance comparison, and 500-fold bootstrap confidence intervals.
6. **Model Diagnostics & Subgroup Heterogeneity** (Page 7)
   - Residual scatter plots, homoskedasticity analysis, and error breakdown across valuation tiers and censoring regimes.
7. **Feature Importance & Interpretability** (Page 8)
   - Permutation importance ranking and partial dependence curves for `MedInc`.
8. **Practical & Business Implications** (Page 9)
   - Operational framework for mortgage underwriting, Automated Valuation Model (AVM) risk gates, ethical considerations, and model limitations.
9. **Conclusions, Recommendations & Academic Bibliography** (Page 10)
   - Seven peer-reviewed citations with verified DOIs.

---

## Dataset Invariants & Provenance

- **Canonical File**: [`data/california_housing.csv`](data/california_housing.csv) (20,640 rows, 8 features, 1 target).
- **Source**: R. Kelley Pace and Ronald P. Barry (1997), *"Sparse spatial autoregressions"*, *Statistics & Probability Letters*, 33(3), 291–297. [DOI: 10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X).
- **Metadata Documentation**: Complete dataset details and variable definitions are documented in [`data/README.md`](data/README.md).
- **Known Invariants Handled**:
  - **Right-Censoring**: `MedHouseVal` capped at 5.0 (\$500,000) for 992 tracts state-wide.
  - **Heavy Tails**: Severe positive skewness in `AveRooms` (max 141.9), `AveOccup` (max 1,243.3), and `Population` (max 35,682) handled robustly via tree-based ensembles without artificial trimming.
  - **Temporal Baseline**: 1990 U.S. Census baseline reflecting historic economic ratios.

---

## Publication Figures

All 9 figures are located in [`reports/figures/`](reports/figures/):

| Figure # | Title | Description |
| :---: | :--- | :--- |
| **01** | `01_target_distribution_and_cap.png` | Histogram & KDE showing the 4.81% censoring spike at \$500,000 |
| **02** | `02_spatial_california_map.png` | Spatial scatter map showing high coastal valuations vs. Central Valley |
| **03** | `03_feature_correlation_heatmap.png` | Pearson correlation matrix highlighting `MedInc` collinearity ($r = 0.69$) |
| **04** | `04_feature_outlier_distributions.png` | Log-scale boxplot distribution illustrating heavy-tailed ratio features |
| **05** | `05_model_performance_comparison.png` | Comparative bar and dot plot contrasting $R^2$, RMSE, and MAE across models |
| **06** | `06_residuals_vs_predicted.png` | Residual scatter plot revealing the upper-diagonal ceiling boundary at \$500k |
| **07** | `07_subgroup_error_analysis.png` | Error distributions across four price tiers and uncapped vs. capped segments |
| **08** | `08_permutation_importance.png` | Permutation feature importance ranking showing dominance of `MedInc` and geography |
| **09** | `09_partial_dependence_medinc.png` | Non-linear partial dependence curve mapping `MedInc` to expected housing valuation |

---

## Repository Architecture

```
housing-price-prediction-capstone/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI running on Python 3.12
├── data/
│   ├── california_housing.csv     # Raw 1990 Census data (20,640 records)
│   └── README.md                  # Dataset provenance, citations, and metadata
├── notebooks/
│   └── capstone.ipynb             # Fully executed end-to-end Jupyter notebook (1.6 MB)
├── report/
│   ├── build_report.py            # ReportLab script generating 10-page formal PDF
│   ├── Capstone_Research_Report.pdf # Formal 10-page Academic Whitepaper
│   └── page_previews/             # Rendered preview PNGs of each PDF page (1-10)
├── reports/
│   ├── figures/                   # 9 publication-grade 150 DPI figures (01 to 09)
│   ├── proof/
│   │   ├── environment.txt        # Pinned virtual environment package manifest
│   │   └── PROOF.md               # Empirical verification matrix and audit log
│   └── results.json               # Serialized experimental results and metrics
├── scripts/
│   ├── build_notebook.py          # Programmatic notebook generator
│   └── run_experiments.py         # Full model training, evaluation, and plotting runner
├── src/
│   ├── __init__.py                # Package declaration
│   ├── data.py                    # Data ingestion, validation, and holdout partitioning
│   ├── evaluation.py              # Metric calculation, bootstrap CI, and subgroup analysis
│   └── models.py                  # Scikit-learn Pipeline definitions and search distributions
├── .gitignore                     # Git ignore rules for Python, virtual environments, and OS files
├── LICENSE                        # MIT License (Mohan Asanthibattula)
├── README.md                      # Project documentation and summary
└── requirements.txt               # Pinned dependencies for reproducible execution
```

---

## Reproduction Instructions

### 1. Environment Setup
Clone the repository and create an isolated Python 3.12 virtual environment:
```bash
git clone https://github.com/mohanasanthibattula-stack/housing-price-prediction-capstone.git
cd housing-price-prediction-capstone

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Figures
Execute the complete empirical pipeline:
```bash
python scripts/run_experiments.py
```
This script downloads/validates `data/california_housing.csv`, splits the data, trains all models, computes metrics, generates all 9 figures in `reports/figures/`, and outputs `reports/results.json`.

### 3. Re-execute the Capstone Notebook
To run the Jupyter notebook end-to-end:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/capstone.ipynb
```

### 4. Build the Research Report PDF
Compile the formal 10-page academic report:
```bash
python report/build_report.py
```
The script will output `report/Capstone_Research_Report.pdf` and verify page count compliance (strictly between 8 and 12 pages).

---

## References

1. **Pace, R. K., & Barry, R. (1997)**. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291–297. [DOI: 10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
2. **Rosen, S. (1974)**. Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy*, 82(1), 34–55. [DOI: 10.1086/260169](https://doi.org/10.1086/260169)
3. **Breiman, L. (2001)**. Random forests. *Machine Learning*, 45(1), 5–32. [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
4. **Friedman, J. H. (2001)**. Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. [DOI: 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
5. **Pedregosa, F., et al. (2011)**. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [URL: https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)
6. **Mullainathan, S., & Spiess, J. (2017)**. Machine learning: An applied econometric approach. *Journal of Economic Perspectives*, 31(2), 87–106. [DOI: 10.1257/jep.31.2.87](https://doi.org/10.1257/jep.31.2.87)
7. **Lundberg, S. M., & Lee, S. I. (2017)**. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774. [URL: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

---

## Author & Academic Integrity

- **Author**: Mohan Asanthibattula
- **Role**: Data Science Intern at RabTech Academy
- **GitHub**: [@mohanasanthibattula-stack](https://github.com/mohanasanthibattula-stack)
- **License**: [MIT License](LICENSE)
