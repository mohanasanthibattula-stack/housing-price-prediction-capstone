# California Housing Dataset Provenance & Metadata

## Overview
This directory contains the canonical raw dataset `california_housing.csv`, recording aggregated demographic and housing statistics for all 20,640 census block groups in California from the 1990 U.S. Census.

## Dataset Specifications
- **File**: `data/california_housing.csv`
- **Observations ($n$)**: 20,640 census block groups
- **Features ($p$)**: 8 continuous predictor variables:
  1. `MedInc`: Median income in block group (in tens of thousands of USD, e.g. 8.3252 = $83,252)
  2. `HouseAge`: Median house age in block group (years)
  3. `AveRooms`: Average number of rooms per household
  4. `AveBedrms`: Average number of bedrooms per household
  5. `Population`: Block group population (number of residents)
  6. `AveOccup`: Average number of household members (occupancy)
  7. `Latitude`: Block group centroid latitude (degrees north)
  8. `Longitude`: Block group centroid longitude (degrees west)
- **Target Variable**: `MedHouseVal`: Median house value for households within a block group (in units of $100,000). For monetary reporting, values are multiplied by $100,000 to convert to US Dollars ($).

## Known Data Characteristics & Domain Invariants
1. **Target Censoring / Capping**: `MedHouseVal` is upper-censored at 5.0 ($500,000) for 965 blocks (4.68% of the dataset), creating an artificial boundary where true values exceeding $500,000 are recorded as 5.0.
2. **Extreme Outliers**: Several ratio features contain extreme heavy-tailed outliers reflecting atypical or unpopulated census blocks (e.g. `AveRooms` reaching 141.9, `AveOccup` reaching 1,243.3, and `Population` up to 35,682). Tree-based ensembles are naturally robust to monotonic feature transformations and extreme tails.
3. **Temporal Invariant**: The dataset reflects 1990 economic, demographic, and geographic conditions. Models trained on this dataset represent historical patterns and are not calibrated for current nominal housing market valuations without temporal and inflation adjustments.

## Scientific Source & Citation
- **Authors**: R. Kelley Pace and Ronald P. Barry (1997)
- **Title**: "Sparse spatial autoregressions"
- **Journal**: *Statistics & Probability Letters*, 33(3), 291–297
- **DOI**: [10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
- **Distribution**: Bundled in `scikit-learn` via `sklearn.datasets.fetch_california_housing()`.

## License
Public Domain / CC0 (1990 U.S. Census Bureau public records).
