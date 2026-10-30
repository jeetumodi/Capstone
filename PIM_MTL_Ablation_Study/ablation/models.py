"""
Neural Network Architectures for PIM-MTL Ablation Study.

Implements configurable multi-task neural networks with flexible activation functions
(GELU, ReLU, LeakyReLU), normalization schemes (SafeBatchNorm1d, standard BatchNorm1d, LayerNorm),
and dynamic input dimensionality.
"""

import os
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation.config import ModelConfig


class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d subclass that handles single-sample batches (N=1) safely
    without raising PyTorch's batch size exception.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )
        return super().forward(x)


def get_activation(act_name: str) -> nn.Module:
    """Factory for activation layers."""
    act_lower = act_name.lower().strip()
    if act_lower == "gelu":
        return nn.GELU()
    elif act_lower == "relu":
        return nn.ReLU()
    elif act_lower == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unsupported activation function: '{act_name}'. Choose from ['gelu', 'relu', 'leaky_relu'].")


def get_normalization(norm_name: str, dim: int) -> nn.Module:
    """Factory for normalization layers."""
    norm_lower = norm_name.lower().strip()
    if norm_lower == "safebatchnorm":
        return SafeBatchNorm1d(dim)
    elif norm_lower == "batchnorm":
        return nn.BatchNorm1d(dim)
    elif norm_lower == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_lower in ("none", "identity"):
        return nn.Identity()
    else:
        raise ValueError(
            f"Unsupported normalization: '{norm_name}'. Choose from ['safebatchnorm', 'batchnorm', 'layernorm', 'none']."
        )


class PIM_MTL_Net(nn.Module):
    """
    Modular Physics-Informed Masked Multi-Task Neural Network.

    Architecture:
        x (Input Features, e.g. 19 or 17 dims)
            |
            v
        Linear(in_dim, 128) -> Norm(128) -> Act -> Dropout(0.2)
            |
            v
        Linear(128, 64) -> Norm(64) -> Act
            |
            v
        Linear(64, 32)
            |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
         Cs Head             Ts Head             Fs Head
         Linear(32, 1)       Linear(32, 1)       Linear(32, 1)
            |                   |                   |
            v                   v                   v
           Cs                  Ts                  Fs
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ModelConfig] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        normalization: str = "safebatchnorm",
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_dims = config.hidden_dims
            dropout = config.dropout
            activation = config.activation
            normalization = config.normalization

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.activation_name = activation
        self.normalization_name = normalization

        # Shared Feature Representation Trunk
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = get_normalization(normalization, hidden_dims[0])
        self.act1 = get_activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = get_normalization(normalization, hidden_dims[1])
        self.act2 = get_activation(activation)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Task-Specific Regression Heads
        self.head_cs = nn.Linear(hidden_dims[2], 1)
        self.head_ts = nn.Linear(hidden_dims[2], 1)
        self.head_fs = nn.Linear(hidden_dims[2], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for stable gradient propagation."""
        nonlinearity = "relu" if self.activation_name in ("relu", "gelu") else "leaky_relu"
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input feature tensor of shape (batch_size, input_dim).
        Returns:
            Predictions tensor of shape (batch_size, 3) representing [Cs, Ts, Fs].
        """
        h = self.drop1(self.act1(self.bn1(self.fc1(x))))
        h = self.act2(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        y_cs = self.head_cs(h)
        y_ts = self.head_ts(h)
        y_fs = self.head_fs(h)

        return torch.cat([y_cs, y_ts, y_fs], dim=-1)

    def save(self, filepath: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save model state dictionary and architecture configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "normalization": self.normalization_name,
            "extra": extra or {},
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu") -> "PIM_MTL_Net":
        """Instantiate and load model from saved checkpoint."""
        payload = torch.load(filepath, map_location=map_location, weights_only=False)
        model = cls(
            input_dim=payload["input_dim"],
            hidden_dims=payload.get("hidden_dims", [128, 64, 32]),
            dropout=payload.get("dropout", 0.2),
            activation=payload.get("activation", "gelu"),
            normalization=payload.get("normalization", "safebatchnorm"),
        )
        model.load_state_dict(payload["state_dict"])
        return model

