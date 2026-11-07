"""
Model Evaluation and Benchmarking Metrics Module.
Calculates RMSE, MAE, MAPE, R2, and Directional Price Trend Forecasting Accuracy.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lag: Optional[np.ndarray] = None,
    trend_tolerance: float = 0.045
) -> Dict[str, float]:
    """
    Compute comprehensive regression and market trend forecasting metrics.
    
    Parameters:
    - y_true: Array of actual modal prices.
    - y_pred: Array of model-forecasted modal prices.
    - y_lag: Optional array of previous period modal prices (used for directional trend accuracy).
    - trend_tolerance: Market noise threshold (default 4.5% for agricultural mandi grade spread).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Mean Absolute Percentage Error (MAPE)
    nonzero_mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    
    results = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }
    
    # Directional Price Trend Forecasting Accuracy across regional categories
    if y_lag is not None:
        y_lag = np.asarray(y_lag).ravel()
        actual_delta = y_true - y_lag
        predicted_delta = y_pred - y_lag
        
        # Trend is correctly identified if signs match or if change is within normal mandi grade tolerance
        tol_band = trend_tolerance * y_lag
        correct_direction = (
            (np.sign(actual_delta) == np.sign(predicted_delta)) |
            (np.abs(actual_delta) < tol_band)
        )
        directional_accuracy = float(np.mean(correct_direction) * 100)
        results["directional_trend_accuracy"] = round(directional_accuracy, 2)
        
    return results


def benchmark_regional_categories(
    df_eval: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    lag_col: str = "price_lag_1w"
) -> pd.DataFrame:
    """
    Break down model forecasting performance across regional crop categories.
    """
    rows = []
    
    for (region, category), group in df_eval.groupby(["region", "category"]):
        metrics = calculate_metrics(
            group[actual_col].values,
            group[pred_col].values,
            group[lag_col].values if lag_col in group.columns else None
        )
        rows.append({
            "region": region,
            "category": category,
            "sample_count": len(group),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
            "directional_accuracy": metrics.get("directional_trend_accuracy", np.nan)
        })
        
    return pd.DataFrame(rows).sort_values(by=["region", "category"]).reset_index(drop=True)
