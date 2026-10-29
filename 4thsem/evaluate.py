"""
Model Evaluation Module.
Computes standard regression metrics on test sets:
- R2 (Coefficient of Determination)
- Adjusted R2: 1 - ((1 - R2) * (n - 1)) / (n - p - 1)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error)
Formats results into readable tables and pandas DataFrames.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> Dict[str, float]:
    """
    Computes R2, Adjusted R2, MAE, RMSE, and MSE for predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    n_features : int
        Number of predictor features (p) for Adjusted R2.

    Returns
    -------
    dict of metric name -> value
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    
    n_samples = len(y_t)
    
    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    
    # Adjusted R2 formula: 1 - ((1 - R2) * (n - 1)) / (n - p - 1)
    if n_samples > n_features + 1:
        adj_r2 = float(1.0 - ((1.0 - r2) * (n_samples - 1.0) / (n_samples - n_features - 1.0)))
    else:
        adj_r2 = r2
        
    return {
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generates predictions from model on X_test and calculates metrics against y_test.
    """
    y_pred = model.predict(X_test)
    if n_features is None:
        n_features = X_test.shape[1] if hasattr(X_test, "shape") else 8
        
    return calculate_metrics(y_test, y_pred, n_features=n_features)


def format_results_table(results_list: list) -> pd.DataFrame:
    """
    Converts list of result dictionaries into a clean, formatted Pandas DataFrame.
    """
    df = pd.DataFrame(results_list)
    cols = ["Target", "Model", "R2", "Adj_R2", "RMSE", "MAE", "MSE"]
    available_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[available_cols]
    return df

