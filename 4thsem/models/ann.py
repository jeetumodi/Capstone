"""
PyTorch Artificial Neural Network (ANN) implementation.
Architecture:
4 Fully Connected Layer blocks with decreasing widths (H -> H/2 -> H/4 -> 1).
Each block includes:
- Linear -> BatchNorm1d -> ReLU -> Dropout(p)
Final layer:
- Linear(H/4, 1) producing scalar strength output.

Training Config:
- MSE Loss
- Adam optimizer (lr=1e-3, weight_decay=1e-4)
- CosineAnnealingLR scheduler
- Batch size = 64
- Early Stopping with patience = 25 epochs based on validation loss.
Wrapped in an sklearn-compatible estimator for seamless Pipeline and Grid integration.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin


class ConcreteANNModule(nn.Module):
    """
    4-Block Fully Connected Feedforward Neural Network for Strength Prediction.
    """
    def __init__(self, in_features: int, H: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.H = H
        self.dropout_p = dropout_p
        
        # Block 1: in_features -> H
        self.block1 = nn.Sequential(
            nn.Linear(in_features, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        # Block 2: H -> H / 2
        self.block2 = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.BatchNorm1d(H // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        # Block 3: H / 2 -> H / 4
        self.block3 = nn.Sequential(
            nn.Linear(H // 2, H // 4),
            nn.BatchNorm1d(H // 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        # Block 4: H / 4 -> 1
        self.out_layer = nn.Linear(H // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out_layer(x)


class PyTorchANNRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for the PyTorch ConcreteANN model.
    Supports early stopping, CosineAnnealingLR, Adam, and hyperparameter tuning.
    """
    def __init__(
        self,
        H: int = 128,
        dropout_p: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 25,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.H = H
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        target_device = self._resolve_device()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples, n_features = X_arr.shape
        self.in_features_ = n_features

        # Split into training and internal validation for Early Stopping
        val_size = int(n_samples * self.val_fraction)
        val_size = max(val_size, 1)
        indices = np.random.permutation(n_samples)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train_t = torch.tensor(X_arr[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y_arr[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(X_arr[val_idx], dtype=torch.float32).to(target_device)
        y_val_t = torch.tensor(y_arr[val_idx], dtype=torch.float32).to(target_device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=(len(train_ds) > self.batch_size),
        )

        # Initialize network
        self.model_ = ConcreteANNModule(
            in_features=n_features,
            H=self.H,
            dropout_p=self.dropout_p,
        ).to(target_device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict = None
        self.train_loss_history_ = []
        self.val_loss_history_ = []

        for epoch in range(self.max_epochs):
            # Training phase
            self.model_.train()
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(target_device)
                batch_y = batch_y.to(target_device)

                optimizer.zero_grad()
                preds = self.model_(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            scheduler.step()
            epoch_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            self.train_loss_history_.append(epoch_train_loss)

            # Validation phase for Early Stopping
            self.model_.eval()
            with torch.no_grad():
                val_preds = self.model_(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
            self.val_loss_history_.append(val_loss)

            # Check for best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.stopped_epoch_ = epoch + 1
                    break
        else:
            self.stopped_epoch_ = self.max_epochs

        # Restore best model weights
        if best_state_dict is not None:
            self.model_.load_state_dict({k: v.to(target_device) for k, v in best_state_dict.items()})

        self.best_val_loss_ = best_val_loss
        self.device_ = target_device
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        target_device = self._resolve_device()
        self.model_.to(target_device)
        self.model_.eval()

        with torch.no_grad():
            tensor_x = torch.tensor(X_arr, dtype=torch.float32, device=target_device)
            preds = self.model_(tensor_x).squeeze(-1).cpu().numpy()

        return np.asarray(preds, dtype=np.float64)

    def save_checkpoint(self, path: str):
        """Save PyTorch model state_dict and hyperparameters."""
        checkpoint = {
            "state_dict": {k: v.cpu() for k, v in self.model_.state_dict().items()},
            "in_features": self.in_features_,
            "H": self.H,
            "dropout_p": self.dropout_p,
            "best_val_loss": self.best_val_loss_,
            "stopped_epoch": getattr(self, "stopped_epoch_", None),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load PyTorch model from saved checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.H = checkpoint["H"]
        self.dropout_p = checkpoint["dropout_p"]
        self.in_features_ = checkpoint["in_features"]
        self.best_val_loss_ = checkpoint["best_val_loss"]
        self.model_ = ConcreteANNModule(
            in_features=self.in_features_,
            H=self.H,
            dropout_p=self.dropout_p,
        )
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.device_ = self._resolve_device()
        self.model_.to(self.device_)
        return self

    def __getstate__(self):
        """Allow pickling by cloning model state dict to CPU."""
        state = self.__dict__.copy()
        if "model_" in state:
            cpu_model = copy.deepcopy(state["model_"]).cpu()
            state["model_"] = cpu_model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "model_"):
            target_device = self._resolve_device()
            self.model_.to(target_device)
            self.device_ = target_device

