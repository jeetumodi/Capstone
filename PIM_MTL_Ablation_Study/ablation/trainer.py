"""
Training Engine for PIM-MTL Ablation Experiments.

Handles:
- Deterministic seeding across Python, NumPy, and PyTorch (CPU and CUDA).
- Gradient-clipped mini-batch optimization with Adam and CosineAnnealingLR.
- Epoch-level loss convergence tracking (Total Loss, Data MSE, Physics Penalty).
- Execution timing and checkpoint management.
"""

import time
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ablation.config import ExperimentConfig
from ablation.models import PIM_MTL_Net
from ablation.losses import PIM_MTL_Loss


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_run(
    config: ExperimentConfig,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    preprocessor: Dict[str, Any],
    seed: int,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[PIM_MTL_Net, List[Dict[str, float]], float]:
    """
    Execute a single training run with the given configuration and initialization seed.

    Returns:
        model: Trained PyTorch network
        history: List of epoch loss records
        duration_sec: Execution wall-clock time in seconds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    X_train, y_train, mask_train, _ = train_data
    input_dim = X_train.shape[1]

    # Instantiate model and loss
    model = PIM_MTL_Net(input_dim=input_dim, config=config.model).to(device)
    criterion = PIM_MTL_Loss(
        config=config.loss,
        target_centers=preprocessor["target_centers"],
        target_scales=preprocessor["target_scales"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.train.min_lr,
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
    )

    history: List[Dict[str, float]] = []
    start_time = time.time()

    model.train()
    for epoch in range(1, config.train.epochs + 1):
        total_loss, total_data, total_phys = 0.0, 0.0, 0.0
        n_samples = len(X_train)

        for bx, by, bm in loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)

            optimizer.zero_grad()
            preds = model(bx)
            loss, l_data, l_phys = criterion(preds, by, bm)
            loss.backward()

            if config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.max_grad_norm)

            optimizer.step()

            bsize = len(bx)
            total_loss += loss.item() * bsize
            total_data += l_data.item() * bsize
            total_phys += l_phys.item() * bsize

        scheduler.step()

        avg_loss = total_loss / n_samples
        avg_data = total_data / n_samples
        avg_phys = total_phys / n_samples
        current_lr = scheduler.get_last_lr()[0]

        record = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "data_loss": avg_data,
            "phys_loss": avg_phys,
            "lr": current_lr,
        }
        history.append(record)

        if verbose and (epoch % 15 == 0 or epoch == 1 or epoch == config.train.epochs):
            print(
                f"  [Seed {seed}] Epoch {epoch:03d}/{config.train.epochs:03d} | "
                f"Loss: {avg_loss:.4f} | Data MSE: {avg_data:.4f} | Phys: {avg_phys:.6f}"
            )

    duration_sec = time.time() - start_time
    return model, history, duration_sec

