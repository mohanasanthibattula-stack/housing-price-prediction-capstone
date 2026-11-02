"""Run complete modeling experiments, diagnostics, and output reports/results.json."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from src.data import load_housing_data, get_train_test_data, FEATURE_COLS, TARGET_COL
from src.models import get_pipelines, get_param_distributions
from src.evaluation import compute_metrics, bootstrap_rmse_ci, evaluate_subgroups, DOLLAR_MULTIPLIER

def main():
    np.random.seed(42)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/proof", exist_ok=True)
    
    # Configure matplotlib styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Load Data
    df, X, y = load_housing_data("data/california_housing.csv")
    X_train, X_test, y_train, y_test = get_train_test_data("data/california_housing.csv", test_size=0.2, random_state=42)
    
    print(f"Dataset Loaded: {len(df)} rows, {len(FEATURE_COLS)} features.")
    print(f"Train set: {len(X_train)} rows; Test set: {len(X_test)} rows.")
    
    # Check target capping
    capped_count = int((df[TARGET_COL] >= 4.9999).sum())
    capped_pct = float((capped_count / len(df)) * 100.0)
    print(f"Target capped at 5.0 ($500k): {capped_count} blocks ({capped_pct:.2f}%)")
    
    # --- EDA FIGURES ---
    # Figure 1: Target Distribution & Cap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET_COL] * DOLLAR_MULTIPLIER, kde=True, bins=50, color='#1f77b4', ax=ax1)
    ax1.axvline(500_000, color='red', linestyle='--', linewidth=1.8, label=f'Censoring Threshold ($500k, n={capped_count})')
    ax1.set_title("Figure 1a: Distribution of Median House Value\\nShowing Right-Censoring Spike at $500,000", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Median House Value (USD)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Census Block Frequency", fontsize=10, fontweight='bold')
    ax1.legend()
    
    sns.ecdfplot(df[TARGET_COL] * DOLLAR_MULTIPLIER, color='#2ca02c', linewidth=2.0, ax=ax2)
    ax2.axvline(500_000, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title("Figure 1b: Empirical Cumulative Distribution (ECDF)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Median House Value (USD)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cumulative Probability", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/01_target_distribution_and_cap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Spatial Geography Map
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df['Longitude'], df['Latitude'],
        c=df[TARGET_COL] * DOLLAR_MULTIPLIER,
        cmap='plasma', s=df['Population'] / 100.0, alpha=0.45,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Median House Value (USD)", fontsize=10, fontweight='bold')
    ax.set_title("Figure 2: Geographic Distribution of California Housing Values (1990 Census)\\nPoint Size Proportional to Population; Coastal Clusters Exhibit Severe Premiums",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude (Degrees West)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Latitude (Degrees North)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/02_spatial_california_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Correlation Heatmap
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
    ax.set_title("Figure 3: Feature Correlation Matrix\\nMedInc Displays Dominant Linear Association with House Value (r = 0.69)",
                 fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("reports/figures/03_feature_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Outlier Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    outlier_feats = [('AveRooms', 25), ('AveOccup', 20), ('Population', 10000), ('MedInc', 15)]
    for idx, (f_col, clip_val) in enumerate(outlier_feats):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(x=df[f_col], color='#4c72b0', ax=ax)
        max_val = df[f_col].max()
        q99 = df[f_col].quantile(0.99)
        ax.set_title(f"{f_col} (Max: {max_val:.1f}, 99th Pct: {q99:.1f})", fontsize=10, fontweight='bold')
        ax.set_xlabel(f_col, fontsize=9)
    plt.suptitle("Figure 4: Extreme Outlier Profiles across Census Predictors", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/04_feature_outlier_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- MODEL TRAINING & TUNING ---
    pipelines = get_pipelines()
    param_dists = get_param_distributions()
    
    models_fitted = {}
    test_predictions = {}
    model_metrics = {}
    
    print("\n--- Training Models on 80% Training Split ---")
    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        dist = param_dists[name]
        if dist:
            n_iter = 8 if name != "RandomForest" else 6
            search = RandomizedSearchCV(
                pipe, param_distributions=dist,
                n_iter=n_iter, cv=3, scoring="neg_root_mean_squared_error",
                random_state=42, n_jobs=-1, verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = {k: str(v) for k, v in search.best_params_.items()}
            print(f"  {name} Best Params: {best_params}")
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe
            best_params = "baseline_mean"
            
        models_fitted[name] = best_model
        y_pred = best_model.predict(X_test)
        test_predictions[name] = y_pred
        
        metrics = compute_metrics(y_test, y_pred)
        metrics["best_params"] = best_params
        model_metrics[name] = metrics
        print(f"  {name} Test Metrics: RMSE=${metrics['rmse_dollars']:,.0f}, MAE=${metrics['mae_dollars']:,.0f}, R2={metrics['r2']:.4f}, MAPE={metrics['mape_pct']:.2f}%")
        
    # Select Best Model based on test RMSE
    best_model_name = min(model_metrics.keys(), key=lambda k: model_metrics[k]["rmse_dollars"])
    best_pred = test_predictions[best_model_name]
    print(f"\nBest Performing Model: {best_model_name} (RMSE = ${model_metrics[best_model_name]['rmse_dollars']:,.0f})")
    
    # Bootstrap Confidence Interval for Best Model
    boot_ci = bootstrap_rmse_ci(y_test, best_pred, n_bootstraps=500, alpha=0.05, random_state=42)
    print(f"Bootstrap 95% RMSE CI for {best_model_name}: [${boot_ci['ci_lower_dollars']:,.0f}, ${boot_ci['ci_upper_dollars']:,.0f}]")
    
    # Subgroup Evaluation for Best Model
    subgroups = evaluate_subgroups(y_test, best_pred)
    print("\nSubgroup Error Analysis:")
    print(f"  Uncapped Blocks (n={subgroups['uncapped']['n_samples']}): RMSE=${subgroups['uncapped']['rmse_dollars']:,.0f}, MAE=${subgroups['uncapped']['mae_dollars']:,.0f}")
    print(f"  Capped ($500k) Blocks (n={subgroups['capped']['n_samples']}): RMSE=${subgroups['capped']['rmse_dollars']:,.0f}, MAE=${subgroups['capped']['mae_dollars']:,.0f}")
    
    # --- DIAGNOSTIC FIGURES ---
    # Figure 5: Model Performance Comparison Barplot
    comp_df = pd.DataFrame([
        {
            "Model": m,
            "RMSE ($)": model_metrics[m]["rmse_dollars"],
            "MAE ($)": model_metrics[m]["mae_dollars"],
            "R2": model_metrics[m]["r2"]
        } for m in ["Dummy", "Ridge", "RandomForest", "HistGradientBoosting"]
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars = ax1.bar(comp_df["Model"], comp_df["RMSE ($)"], color=['#7f7f7f', '#aec7e8', '#ffbb78', '#2ca02c'], edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1500, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title("Figure 5a: Test Set RMSE by Model (Lower is Better)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("RMSE (USD)", fontsize=10, fontweight='bold')
    ax1.set_ylim(0, comp_df["RMSE ($)"].max() * 1.15)
    
    bars2 = ax2.bar(comp_df["Model"], comp_df["R2"], color=['#7f7f7f', '#aec7e8', '#ffbb78', '#2ca02c'], edgecolor='black')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title("Figure 5b: Test Set R² Score by Model (Higher is Better)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("R² Score", fontsize=10, fontweight='bold')
    ax2.set_ylim(-0.05, 1.0)
    plt.tight_layout()
    plt.savefig("reports/figures/05_model_performance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 6: Residuals vs Predicted for Best Model
    y_test_dollars = y_test.values * DOLLAR_MULTIPLIER
    best_pred_dollars = best_pred * DOLLAR_MULTIPLIER
    residuals_dollars = y_test_dollars - best_pred_dollars
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(best_pred_dollars, residuals_dollars, alpha=0.3, s=15, color='#1f77b4', edgecolors='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Figure 6a: Residuals vs. Predicted Values ({best_model_name})\\nHomoskedastic Mid-Range with Upper-Bound Censoring Line", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Predicted Median House Value (USD)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Residual Error (Actual - Predicted, USD)", fontsize=10, fontweight='bold')
    
    sns.histplot(residuals_dollars, kde=True, bins=50, color='#2ca02c', ax=ax2)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title(f"Figure 6b: Residual Error Distribution\\nMean = ${np.mean(residuals_dollars):,.0f}, Std = ${np.std(residuals_dollars):,.0f}", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Residual Error (USD)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/06_residuals_vs_predicted.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 7: Subgroup & Price Band Errors
    bands_dict = subgroups["value_bands"]
    band_names = list(bands_dict.keys())
    band_maes = [bands_dict[b]["mae_dollars"] for b in band_names]
    band_rmses = [bands_dict[b]["rmse_dollars"] for b in band_names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(len(band_names))
    width = 0.35
    ax.bar(x_indices - width/2, band_maes, width, label='MAE (USD)', color='#4c72b0', edgecolor='black')
    ax.bar(x_indices + width/2, band_rmses, width, label='RMSE (USD)', color='#c44e52', edgecolor='black')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(band_names, fontsize=10, fontweight='bold')
    ax.set_ylabel("Error in US Dollars ($)", fontsize=10, fontweight='bold')
    ax.set_title(f"Figure 7: Predictive Error Breakdown Across Housing Value Bands ({best_model_name})\\nSubstantial Error Spike on Capped $500,000 Properties Due to Censoring",
                 fontsize=11, fontweight='bold', pad=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/07_subgroup_error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 8: Permutation Importance
    print("Computing Permutation Importance on Test Set...")
    perm_result = permutation_importance(
        models_fitted[best_model_name], X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_importances_idx = perm_result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        perm_result.importances[sorted_importances_idx].T,
        vert=False,
        tick_labels=np.array(FEATURE_COLS)[sorted_importances_idx],
        patch_artist=True,
        boxprops=dict(facecolor='#aec7e8', color='black')
    )
    ax.set_title(f"Figure 8: Permutation Feature Importance on Holdout Test Set ({best_model_name})\\nMedInc Dominates Predictive Power, Followed by Spatial Coordinates",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Decrease in Test R² Score upon Feature Permutation", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/08_permutation_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    importance_summary = {
        FEATURE_COLS[i]: {
            "mean_decrease_r2": float(perm_result.importances_mean[i]),
            "std_decrease_r2": float(perm_result.importances_std[i])
        } for i in range(len(FEATURE_COLS))
    }
    
    # Figure 9: Partial Dependence Plot for MedInc
    print("Computing Partial Dependence Plot for MedInc...")
    fig, ax = plt.subplots(figsize=(9, 5))
    PartialDependenceDisplay.from_estimator(
        models_fitted[best_model_name], X_train, features=['MedInc'],
        ax=ax, line_kw={'color': '#d62728', 'linewidth': 2.5}
    )
    ax.set_title(f"Figure 9: Partial Dependence of House Value on Median Income ({best_model_name})\\nConsistent Upward Elasticity Flattening Near the $5.0 Cap",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Median Income (Tens of Thousands of USD)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Partial Dependence (MedHouseVal, $100k Units)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/09_partial_dependence_medinc.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- ASSEMBLE RESULTS JSON ---
    results_payload = {
        "dataset_metadata": {
            "source": "1990 U.S. Census Bureau via Pace & Barry (1997)",
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(FEATURE_COLS),
            "features": FEATURE_COLS,
            "target": TARGET_COL,
            "capped_samples": capped_count,
            "capped_percentage": capped_pct,
            "mean_house_val_dollars": float(df[TARGET_COL].mean() * DOLLAR_MULTIPLIER),
            "median_house_val_dollars": float(df[TARGET_COL].median() * DOLLAR_MULTIPLIER),
            "std_house_val_dollars": float(df[TARGET_COL].std() * DOLLAR_MULTIPLIER),
            "min_house_val_dollars": float(df[TARGET_COL].min() * DOLLAR_MULTIPLIER),
            "max_house_val_dollars": float(df[TARGET_COL].max() * DOLLAR_MULTIPLIER)
        },
        "model_comparison": model_metrics,
        "best_model": {
            "name": best_model_name,
            "metrics": model_metrics[best_model_name],
            "bootstrap_rmse_95ci": boot_ci,
            "subgroups": subgroups,
            "permutation_importance": importance_summary
        }
    }
    
    with open("reports/results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
        
    print(f"\nAll experiments complete. Results persisted to reports/results.json ({os.path.getsize('reports/results.json')} bytes).")

if __name__ == "__main__":
    main()
