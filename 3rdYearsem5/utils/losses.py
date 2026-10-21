"""
Physics-Informed Masked Multi-Task Loss for ConcreteXAI.

Combines masked empirical MSE across partially observed mechanical properties
(Cs, Ts, Fs) with a differentiable physics-informed penalty over strength ratios.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def PIM_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    target_scales: Optional[torch.Tensor] = None,
    target_centers: Optional[torch.Tensor] = None,
    task_weights: Optional[torch.Tensor] = None,
    lambda_phys: float = 0.5,
    rt_bounds: Tuple[float, float] = (0.06, 0.18),
    rf_bounds: Tuple[float, float] = (0.08, 0.25),
    epsilon: float = 1e-6,
    normalization: str = "per_task",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional implementation of the Physics-Informed Masked Multi-Task Loss.

    Args:
        y_pred: Model predictions of shape (N, 3) in order [Cs, Ts, Fs].
        y_true: Ground truth targets of shape (N, 3).
        mask: Binary observation mask of shape (N, 3) (1.0 = observed, 0.0 = missing).
        target_scales: Scaling factor tensor of shape (3,) for denormalization to MPa.
        target_centers: Center/shift tensor of shape (3,) for denormalization to MPa.
        task_weights: Weight tensor of shape (3,) for [Cs, Ts, Fs]. Defaults to [1.0, 1.2, 1.5].
        lambda_phys: Weight coefficient for the physics penalty term.
        rt_bounds: Physical ratio bounds (min, max) for Rt = Ts / Cs.
        rf_bounds: Physical ratio bounds (min, max) for Rf = Fs / Cs.
        epsilon: Small numerical stabilizer to prevent division by zero.
        normalization: 'per_task' (normalizes each task by observed count) or 'batch_size'.

    Returns:
        Tuple of (loss_total, loss_data, loss_phys).
    """
    device = y_pred.device
    N = y_pred.size(0)

    if task_weights is None:
        task_weights = torch.tensor([1.0, 1.2, 1.5], dtype=torch.float32, device=device)
    else:
        task_weights = task_weights.to(device)

    # 1. Masked Empirical Data Loss
    squared_errors = (y_pred - y_true) ** 2
    masked_errors = mask * squared_errors

    if normalization == "per_task":
        observed_counts = torch.clamp(torch.sum(mask, dim=0), min=1.0)
        task_mse = torch.sum(masked_errors, dim=0) / observed_counts
        loss_data = torch.sum(task_weights * task_mse)
    else:
        weighted_errors = masked_errors * task_weights.unsqueeze(0)
        loss_data = torch.sum(weighted_errors) / max(1, N)

    # 2. Differentiable Physics Ratio Regularization
    if target_scales is not None and target_centers is not None:
        y_orig = y_pred * target_scales.to(device).unsqueeze(0) + target_centers.to(device).unsqueeze(0)
    else:
        y_orig = y_pred

    # Concrete compressive strength is non-negative and physically > 1.0 MPa
    cs = torch.clamp(y_orig[:, 0], min=1.0)
    ts = torch.clamp(y_orig[:, 1], min=0.0)
    fs = torch.clamp(y_orig[:, 2], min=0.0)

    rt = ts / cs
    rf = fs / cs

    viol_rt_upper = F.relu(rt - rt_bounds[1]) ** 2
    viol_rt_lower = F.relu(rt_bounds[0] - rt) ** 2
    viol_rf_upper = F.relu(rf - rf_bounds[1]) ** 2
    viol_rf_lower = F.relu(rf_bounds[0] - rf) ** 2

    loss_phys = torch.mean(viol_rt_upper + viol_rt_lower + viol_rf_upper + viol_rf_lower)

    # 3. Total Loss
    loss_total = loss_data + lambda_phys * loss_phys
    return loss_total, loss_data, loss_phys


class PIM_MTL_Loss(nn.Module):
    """
    PyTorch Module wrapper for Physics-Informed Masked Multi-Task Loss.
    Maintains registered buffers for target scaling parameters and task weights.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        lambda_phys: float = 0.5,
        rt_bounds: Tuple[float, float] = (0.06, 0.18),
        rf_bounds: Tuple[float, float] = (0.08, 0.25),
        normalization: str = "per_task",
        epsilon: float = 1e-6,
        target_centers: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
        target_scales: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if task_weights is None:
            task_weights = {"Cs": 1.0, "Ts": 1.2, "Fs": 1.5}

        self.lambda_phys = float(lambda_phys)
        self.rt_bounds = rt_bounds
        self.rf_bounds = rf_bounds
        self.normalization = normalization
        self.epsilon = float(epsilon)

        w_tensor = torch.tensor(
            [task_weights["Cs"], task_weights["Ts"], task_weights["Fs"]],
            dtype=torch.float32,
        )
        self.register_buffer("weights", w_tensor)

        if target_centers is not None and target_scales is not None:
            self.register_buffer("target_centers", torch.as_tensor(target_centers, dtype=torch.float32))
            self.register_buffer("target_scales", torch.as_tensor(target_scales, dtype=torch.float32))
            self.has_scaling = True
        else:
            self.register_buffer("target_centers", torch.zeros(3, dtype=torch.float32))
            self.register_buffer("target_scales", torch.ones(3, dtype=torch.float32))
            self.has_scaling = False

    def set_scalers(self, centers: np.ndarray, scales: np.ndarray) -> None:
        """Update denormalization centers and scales from fitted preprocessor."""
        self.target_centers.copy_(torch.as_tensor(centers, dtype=torch.float32, device=self.target_centers.device))
        self.target_scales.copy_(torch.as_tensor(scales, dtype=torch.float32, device=self.target_scales.device))
        self.has_scaling = True

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scales = self.target_scales if self.has_scaling else None
        centers = self.target_centers if self.has_scaling else None

        return PIM_loss(
            y_pred=y_pred,
            y_true=y_true,
            mask=mask,
            target_scales=scales,
            target_centers=centers,
            task_weights=self.weights,
            lambda_phys=self.lambda_phys,
            rt_bounds=self.rt_bounds,
            rf_bounds=self.rf_bounds,
            epsilon=self.epsilon,
            normalization=self.normalization,
        )
