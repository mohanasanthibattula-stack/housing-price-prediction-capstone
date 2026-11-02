"""Model evaluation metrics and bootstrap statistical functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

DOLLAR_MULTIPLIER = 100_000.0

def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2, and MAPE both in normalized units and US Dollars."""
    rmse_unit = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_unit = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    
    rmse_dollars = rmse_unit * DOLLAR_MULTIPLIER
    mae_dollars = mae_unit * DOLLAR_MULTIPLIER
    
    return {
        "rmse_unit": rmse_unit,
        "mae_unit": mae_unit,
        "r2": r2,
        "mape_pct": mape,
        "rmse_dollars": rmse_dollars,
        "mae_dollars": mae_dollars
    }

def bootstrap_rmse_ci(y_true, y_pred, n_bootstraps=500, alpha=0.05, random_state=42):
    """Compute 95% bootstrap confidence interval for test set RMSE."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    boot_rmse_dollars = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        sample_rmse = np.sqrt(mean_squared_error(y_true_arr[indices], y_pred_arr[indices]))
        boot_rmse_dollars.append(sample_rmse * DOLLAR_MULTIPLIER)
        
    boot_rmse_dollars = np.array(boot_rmse_dollars)
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    
    ci_lower = float(np.percentile(boot_rmse_dollars, lower_pct))
    ci_upper = float(np.percentile(boot_rmse_dollars, upper_pct))
    
    return {
        "n_bootstraps": n_bootstraps,
        "ci_lower_dollars": ci_lower,
        "ci_upper_dollars": ci_upper,
        "mean_bootstrap_rmse_dollars": float(np.mean(boot_rmse_dollars)),
        "std_bootstrap_rmse_dollars": float(np.std(boot_rmse_dollars))
    }

def evaluate_subgroups(y_true, y_pred):
    """Evaluate error performance across value bands and capped (5.0) blocks."""
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
    
    # Capped vs Uncapped
    is_capped = y_true_s >= 4.9999
    
    capped_metrics = compute_metrics(y_true_s[is_capped], y_pred_s[is_capped])
    uncapped_metrics = compute_metrics(y_true_s[~is_capped], y_pred_s[~is_capped])
    
    # Value Bands ($0-$150k, $150k-$300k, $300k-$500k, $500k capped)
    bands = pd.cut(
        y_true_s * DOLLAR_MULTIPLIER,
        bins=[0, 150_000, 300_000, 499_999, 600_000],
        labels=["< $150k", "$150k-$300k", "$300k-$500k", "Capped ($500k)"]
    )
    
    band_results = {}
    for band_name in ["< $150k", "$150k-$300k", "$300k-$500k", "Capped ($500k)"]:
        mask = bands == band_name
        if mask.sum() > 0:
            m = compute_metrics(y_true_s[mask], y_pred_s[mask])
            band_results[band_name] = {
                "n_samples": int(mask.sum()),
                "rmse_dollars": m["rmse_dollars"],
                "mae_dollars": m["mae_dollars"],
                "mape_pct": m["mape_pct"]
            }
            
    return {
        "capped": {
            "n_samples": int(is_capped.sum()),
            "rmse_dollars": capped_metrics["rmse_dollars"],
            "mae_dollars": capped_metrics["mae_dollars"],
            "mape_pct": capped_metrics["mape_pct"]
        },
        "uncapped": {
            "n_samples": int((~is_capped).sum()),
            "rmse_dollars": uncapped_metrics["rmse_dollars"],
            "mae_dollars": uncapped_metrics["mae_dollars"],
            "mape_pct": uncapped_metrics["mape_pct"]
        },
        "value_bands": band_results
    }
