"""
Physical Constraints & ACI 318 Feasibility Auditing Module.
Computes empirical accuracy metrics and audits models for domain compliance:
1. ACI 318 Tensile Ratio Envelope: 0.06 <= Ts / Cs <= 0.18
2. ACI 318 Flexural Ratio Envelope: 0.08 <= Fs / Cs <= 0.25
3. Physically Impossible Predictions: Ts >= Cs
4. Structural Hierarchy Violations: Ts >= Fs
5. Non-negativity: Cs <= 0 or Ts <= 0 or Fs <= 0
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import ACI_BOUNDS


def compute_accuracy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int = 8,
) -> Dict[str, float]:
    """Calculate MSE, RMSE, R2, Adjusted R2, and MAE for observed targets."""
    mask = ~np.isnan(y_true)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n_samples = len(y_t)

    if n_samples == 0:
        return {"MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "Adj_R2": np.nan, "MAE": np.nan, "N": 0}

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))

    if n_samples > n_features + 1:
        adj_r2 = float(1.0 - ((1.0 - r2) * (n_samples - 1.0) / (n_samples - n_features - 1.0)))
    else:
        adj_r2 = r2

    return {
        "N": n_samples,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mae,
    }


def audit_physical_constraints(
    cs_pred: np.ndarray,
    ts_pred: np.ndarray,
    fs_pred: np.ndarray,
    rt_bounds: Tuple[float, float] = (ACI_BOUNDS["Rt_min"], ACI_BOUNDS["Rt_max"]),
    rf_bounds: Tuple[float, float] = (ACI_BOUNDS["Rf_min"], ACI_BOUNDS["Rf_max"]),
) -> Dict[str, Any]:
    """
    Perform a complete physical and building-code constraint audit
    on predicted triplets [Cs, Ts, Fs].
    """
    cs = np.asarray(cs_pred, dtype=np.float64)
    ts = np.asarray(ts_pred, dtype=np.float64)
    fs = np.asarray(fs_pred, dtype=np.float64)
    n_total = len(cs)

    # Stabilize Cs for ratio calculation (clamp min at 1e-4)
    cs_safe = np.maximum(cs, 1e-4)
    rt = ts / cs_safe
    rf = fs / cs_safe

    # 1. Non-negativity Violations
    non_pos_mask = (cs <= 0.0) | (ts <= 0.0) | (fs <= 0.0)
    non_pos_count = int(np.sum(non_pos_mask))
    non_pos_rate = float((non_pos_count / n_total) * 100.0)

    # 2. Physically Impossible: Tensile >= Compressive (Ts >= Cs)
    impossible_mask = (ts >= cs)
    impossible_count = int(np.sum(impossible_mask))
    impossible_rate = float((impossible_count / n_total) * 100.0)
    max_impossible_ratio = float(np.max(rt)) if n_total > 0 else 0.0

    # 3. Strength Hierarchy Violations: Tensile >= Flexural (Ts >= Fs)
    # Using tiny tolerance 1e-5 to avoid numerical floating point noise
    hierarchy_mask = (ts >= fs - 1e-5)
    hierarchy_count = int(np.sum(hierarchy_mask))
    hierarchy_rate = float((hierarchy_count / n_total) * 100.0)

    # 4. ACI 318 Tensile Ratio Envelope [0.06 <= Rt <= 0.18]
    rt_lower_mask = (rt < rt_bounds[0] - 1e-5)
    rt_upper_mask = (rt > rt_bounds[1] + 1e-5)
    rt_viol_mask = rt_lower_mask | rt_upper_mask
    rt_viol_count = int(np.sum(rt_viol_mask))
    rt_viol_rate = float((rt_viol_count / n_total) * 100.0)

    delta_rt = np.maximum(0.0, rt_bounds[0] - rt) + np.maximum(0.0, rt - rt_bounds[1])
    mean_delta_rt = float(np.mean(delta_rt[rt_viol_mask])) if rt_viol_count > 0 else 0.0
    max_delta_rt = float(np.max(delta_rt)) if n_total > 0 else 0.0

    # 5. ACI 318 Flexural Ratio Envelope [0.08 <= Rf <= 0.25]
    rf_lower_mask = (rf < rf_bounds[0] - 1e-5)
    rf_upper_mask = (rf > rf_bounds[1] + 1e-5)
    rf_viol_mask = rf_lower_mask | rf_upper_mask
    rf_viol_count = int(np.sum(rf_viol_mask))
    rf_viol_rate = float((rf_viol_count / n_total) * 100.0)

    delta_rf = np.maximum(0.0, rf_bounds[0] - rf) + np.maximum(0.0, rf - rf_bounds[1])
    mean_delta_rf = float(np.mean(delta_rf[rf_viol_mask])) if rf_viol_count > 0 else 0.0
    max_delta_rf = float(np.max(delta_rf)) if n_total > 0 else 0.0

    # 6. Overall Physical Compliance (Specimens with ZERO violations)
    any_viol_mask = non_pos_mask | impossible_mask | hierarchy_mask | rt_viol_mask | rf_viol_mask
    overall_compliance_count = int(np.sum(~any_viol_mask))
    overall_compliance_rate = float((overall_compliance_count / n_total) * 100.0)
    overall_violation_rate = float((np.sum(any_viol_mask) / n_total) * 100.0)

    return {
        "Total_Specimens": n_total,
        # Impossible violations
        "Impossible_Count": impossible_count,
        "Impossible_Rate_Pct": impossible_rate,
        "Max_Ts_over_Cs_Ratio": max_impossible_ratio,
        # Hierarchy violations
        "Hierarchy_Viol_Count": hierarchy_count,
        "Hierarchy_Viol_Rate_Pct": hierarchy_rate,
        # ACI Tensile Envelope
        "ACI_Rt_Viol_Count": rt_viol_count,
        "ACI_Rt_Viol_Rate_Pct": rt_viol_rate,
        "Mean_Delta_Rt": mean_delta_rt,
        "Max_Delta_Rt": max_delta_rt,
        # ACI Flexural Envelope
        "ACI_Rf_Viol_Count": rf_viol_count,
        "ACI_Rf_Viol_Rate_Pct": rf_viol_rate,
        "Mean_Delta_Rf": mean_delta_rf,
        "Max_Delta_Rf": max_delta_rf,
        # Non-positivity
        "Non_Positive_Count": non_pos_count,
        "Non_Positive_Rate_Pct": non_pos_rate,
        # Overall
        "Overall_Compliant_Count": overall_compliance_count,
        "Overall_Compliance_Rate_Pct": overall_compliance_rate,
        "Overall_Violation_Rate_Pct": overall_violation_rate,
    }
