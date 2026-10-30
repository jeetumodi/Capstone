"""
Evaluation Engine for Predictive and Physical Compliance Metrics.

Calculates:
- Statistical Predictive Metrics (R², RMSE, MAE) per task on genuine observed test targets.
- Physical Feasibility Metrics:
    - Physically Impossible Rate (Ts >= Cs)
    - Hierarchy Inversion Rate (Ts >= Fs)
    - ACI 318 Tensile Ratio Violation Rate (Rt outside [0.06, 0.18])
    - ACI 318 Flexural Ratio Violation Rate (Rf outside [0.08, 0.25])
    - Overall Physical Compliance Rate (PCR, % of specimens satisfying all physical invariants)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_model_on_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    mask_test: np.ndarray,
    preprocessor: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Generate model predictions on test partition and calculate comprehensive
    predictive and physical compliance metrics.

    Returns:
        metrics_dict: Dictionary containing all evaluated scalar metrics.
        preds_raw: Array of shape (N_test, 3) with denormalized predictions in MPa.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_scaled = model(x_tensor).cpu().numpy()

    centers = preprocessor["target_centers"]
    scales = preprocessor["target_scales"]
    preds_raw = preds_scaled * scales + centers

    metrics = compute_all_metrics(preds_raw, y_test_raw, mask_test)
    return metrics, preds_raw


def compute_all_metrics(
    preds_raw: np.ndarray,
    y_test_raw: np.ndarray,
    mask_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute predictive and physical compliance metrics given predictions and ground truth.
    """
    n_test = len(y_test_raw)
    target_names = ["Cs", "Ts", "Fs"]
    metrics: Dict[str, float] = {
        "n_test_total": float(n_test),
    }

    # 1. Predictive Performance per Task (Observed Targets Only)
    r2_list, rmse_list, mae_list = [], [], []
    for i, col in enumerate(target_names):
        valid = (mask_test[:, i] == 1.0)
        n_obs = int(valid.sum())
        metrics[f"{col}_observed_n"] = float(n_obs)
        metrics[f"{col}_coverage_pct"] = float((n_obs / n_test) * 100.0)

        if n_obs > 1:
            y_true = y_test_raw[valid, i]
            y_pred = preds_raw[valid, i]

            r2 = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
        else:
            r2, rmse, mae = float("nan"), float("nan"), float("nan")

        metrics[f"{col}_r2"] = r2
        metrics[f"{col}_rmse"] = rmse
        metrics[f"{col}_mae"] = mae

        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)

    metrics["mean_r2"] = float(np.nanmean(r2_list))
    metrics["mean_rmse"] = float(np.nanmean(rmse_list))
    metrics["mean_mae"] = float(np.nanmean(mae_list))

    # 2. Physical Feasibility & Invariant Audit
    # Numerically safe compressive strength denominator (min 1.0 MPa)
    cs_pred_safe = np.maximum(preds_raw[:, 0], 1.0)
    ts_pred_safe = np.maximum(preds_raw[:, 1], 0.0)
    fs_pred_safe = np.maximum(preds_raw[:, 2], 0.0)

    rt = ts_pred_safe / cs_pred_safe
    rf = fs_pred_safe / cs_pred_safe

    # Invariant 1: Physically Impossible Rate (Ts >= Cs)
    # Concrete tensile strength can never exceed compressive strength
    viol_impossible = (preds_raw[:, 1] >= preds_raw[:, 0])
    pvr_impossible = float(np.mean(viol_impossible) * 100.0)
    metrics["pvr_impossible_rate"] = pvr_impossible
    metrics["compliance_impossible_rate"] = 100.0 - pvr_impossible

    # Invariant 2: Hierarchy Inversion Rate (Ts >= Fs)
    # Flexural strength consistently exceeds splitting tensile strength
    viol_hierarchy = (preds_raw[:, 1] >= preds_raw[:, 2])
    pvr_hierarchy = float(np.mean(viol_hierarchy) * 100.0)
    metrics["pvr_hierarchy_inversion_rate"] = pvr_hierarchy
    metrics["compliance_hierarchy_rate"] = 100.0 - pvr_hierarchy

    # Invariant 3: ACI 318 Tensile Ratio Bounds [0.06 <= Rt <= 0.18]
    viol_rt = (rt < 0.06) | (rt > 0.18)
    pvr_rt = float(np.mean(viol_rt) * 100.0)
    metrics["pvr_tensile_ratio_rate"] = pvr_rt
    metrics["compliance_tensile_ratio_rate"] = 100.0 - pvr_rt

    # Invariant 4: ACI 318 Flexural Ratio Bounds [0.08 <= Rf <= 0.25]
    viol_rf = (rf < 0.08) | (rf > 0.25)
    pvr_rf = float(np.mean(viol_rf) * 100.0)
    metrics["pvr_flexural_ratio_rate"] = pvr_rf
    metrics["compliance_flexural_ratio_rate"] = 100.0 - pvr_rf

    # Invariant 5: Overall Physical Compliance Rate (PCR)
    # Satisfies ALL 4 physical criteria simultaneously
    fully_compliant = (~viol_impossible) & (~viol_hierarchy) & (~viol_rt) & (~viol_rf)
    overall_pcr = float(np.mean(fully_compliant) * 100.0)
    metrics["overall_pcr"] = overall_pcr

    return metrics
