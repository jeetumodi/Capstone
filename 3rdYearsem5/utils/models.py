"""
Neural Network Architectures for Concrete Strength Prediction.

Implements the Physics-Informed Masked Multi-Task Network (PIM_MTL_Net)
with a shared trunk encoder and three independent task heads for [Cs, Ts, Fs].
"""

import os
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PIM_MTL_Net(nn.Module):
    """
    Physics-Informed Masked Multi-Task Neural Network.

    Architecture:
        x (Input Features, e.g. 19 dims)
            |
            v
        Linear(in_dim, 128) -> SafeBatchNorm1d(128) -> GELU -> Dropout(0.2)
            |
            v
        Linear(128, 64) -> SafeBatchNorm1d(64) -> GELU
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
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Shared Feature Representation Trunk
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = SafeBatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = SafeBatchNorm1d(hidden_dims[1]) if use_batch_norm else nn.Identity()
        self.act2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Task-Specific Regression Heads
        self.head_cs = nn.Linear(hidden_dims[2], 1)
        self.head_ts = nn.Linear(hidden_dims[2], 1)
        self.head_fs = nn.Linear(hidden_dims[2], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for stable gradient propagation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
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
        )
        model.load_state_dict(payload["state_dict"])
        return model
