"""
Intelligent Concrete Strength Prediction Pipeline
Physics-Informed Deep Neural Network + ML Baselines + Inductive Conformal Calibration

Targeting: ConcreteXAI Benchmark (Data_train.npz, Data_validation.npz, Data_test.npz)
Feature Semantics:
    inputs[:, 0] -> Design_Fc_Mpa   (Design Compressive Strength, MPa)
    inputs[:, 1] -> Curing_age_days (Specimen Hydration Age, days)
    inputs[:, 2] -> Er_ohm-cm       (Electrical Resistivity, Ohm-cm)
    inputs[:, 3] -> UPV_m/s         (Ultrasonic Pulse Velocity, m/s)
Target:
    targets[:]   -> Cs_Mpa          (Compressive Strength, MPa)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb


# ==============================================================================
# 0. CONFIGURATION & DOMAIN CONSTANTS
# ==============================================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4

FEATURE_NAMES = ["Design_Fc_Mpa", "Curing_age_days", "Er_ohm-cm", "UPV_m/s"]
TARGET_NAME = "Cs_Mpa"

# Exact empirical de-standardization parameters derived from ConcreteXAI/Data.csv
SCALER_MEAN_X = np.array([28.95927602, 40.44343891, 6.76044425, 3808.95926674])
SCALER_STD_X = np.array([3.50864058, 33.30501729, 2.97025360, 560.97765487])
SCALER_MEAN_Y = 31.19311346
SCALER_STD_Y = 9.58773725

# Index of features for physics-informed monotonicity penalties
AGE_IDX = 1       # Curing_age_days
DESIGN_FC_IDX = 0 # Design_Fc_Mpa


# ==============================================================================
# 1. DETERMINISTIC REPRODUCIBILITY & DEVICE SETUP
# ==============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Compute Device: {device}")


# ==============================================================================
# 2. DATA INGESTION & VALIDATION
# ==============================================================================

def resolve_data_path(filename: str) -> str:
    """Finds NPZ file in current directory or inside ConcreteXAI subdirectories."""
    candidates = [
        filename,
        os.path.join("ConcreteXAI", "Model_Code", filename),
        os.path.join("ConcreteXAI", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Unable to locate '{filename}'. Searched paths: {candidates}"
    )


def load_npz_data():
    """Loads and standardizes shape formats for train, val, and test splits."""
    train_path = resolve_data_path("Data_train.npz")
    val_path = resolve_data_path("Data_validation.npz")
    test_path = resolve_data_path("Data_test.npz")

    train_npz = np.load(train_path)
    val_npz = np.load(val_path)
    test_npz = np.load(test_path)

    X_train = train_npz["inputs"].astype(np.float64)
    y_train = train_npz["targets"].astype(np.float64).ravel()

    X_val = val_npz["inputs"].astype(np.float64)
    y_val = val_npz["targets"].astype(np.float64).ravel()

    X_test = test_npz["inputs"].astype(np.float64)
    y_test = test_npz["targets"].astype(np.float64).ravel()

    print("=" * 80)
    print("DATASET LOADED & VALIDATED")
    print("=" * 80)
    print(f"Train split      : X={X_train.shape}, y={y_train.shape}")
    print(f"Validation split : X={X_val.shape}, y={y_val.shape}")
    print(f"Test split       : X={X_test.shape}, y={y_test.shape}")
    print(f"Features         : {FEATURE_NAMES}")
    print(f"Target           : {TARGET_NAME}")
    print("=" * 80)

    # Sanity validations
    assert X_train.shape[1] == 4, f"Expected 4 input features, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Train X and y length mismatch"
    assert len(X_val) == len(y_val), "Val X and y length mismatch"
    assert len(X_test) == len(y_test), "Test X and y length mismatch"
    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isnan(y_train).any(), "NaN found in y_train"

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# 3. METRICS UTILITY (DUAL DOMAIN: STANDARDIZED + REAL MPa)
# ==============================================================================

def to_mpa(y_standardized: np.ndarray) -> np.ndarray:
    """Converts standardized values back to physical MPa units."""
    return y_standardized * SCALER_STD_Y + SCALER_MEAN_Y


def compute_metrics(y_true_std: np.ndarray, y_pred_std: np.ndarray):
    """Computes regression metrics in both standardized and real-world MPa units."""
    r2 = r2_score(y_true_std, y_pred_std)
    rmse_std = np.sqrt(mean_squared_error(y_true_std, y_pred_std))
    mae_std = mean_absolute_error(y_true_std, y_pred_std)

    # Real MPa units
    y_true_mpa = to_mpa(y_true_std)
    y_pred_mpa = to_mpa(y_pred_std)
    rmse_mpa = np.sqrt(mean_squared_error(y_true_mpa, y_pred_mpa))
    mae_mpa = mean_absolute_error(y_true_mpa, y_pred_mpa)
    mape = np.mean(np.abs((y_true_mpa - y_pred_mpa) / (y_true_mpa + 1e-8))) * 100

    return {
        "R2": r2,
        "RMSE_std": rmse_std,
        "MAE_std": mae_std,
        "RMSE_MPa": rmse_mpa,
        "MAE_MPa": mae_mpa,
        "MAPE_%": mape,
    }


# ==============================================================================
# 4. ML BENCHMARKS (RIDGE, RANDOM FOREST, XGBOOST, LIGHTGBM)
# ==============================================================================

def run_ml_baselines(X_train, y_train, X_val, y_val, X_test, y_test):
    """Trains and evaluates classical and gradient boosted regression baselines."""
    print("\n" + "=" * 80)
    print("TRAINING ML BENCHMARKS")
    print("=" * 80)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=SEED, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=SEED, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1
        ),
    }

    results = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"--> Training {name}...")
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        test_predictions[name] = test_preds
        val_metrics = compute_metrics(y_val, val_preds)
        test_metrics = compute_metrics(y_test, test_preds)

        results[name] = {
            "model": model,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        print(
            f"    Test R²: {test_metrics['R2']:.4f} | "
            f"RMSE: {test_metrics['RMSE_MPa']:.3f} MPa | "
            f"MAE: {test_metrics['MAE_MPa']:.3f} MPa | "
            f"MAPE: {test_metrics['MAPE_%']:.2f}%"
        )

    return results, test_predictions


# ==============================================================================
# 5. PHYSICS-INFORMED DEEP NEURAL NETWORK (PINN-STRENGTH)
# ==============================================================================

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)). Smooth non-monotonic profile."""
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class ConcretePINN(nn.Module):
    """
    Deep Residual MLP with Mish activations and Monte Carlo Dropout
    for robust point prediction and epistemic uncertainty quantification.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, dropout_rate: float = 0.15):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Mish(),
            nn.Dropout(dropout_rate)
        )

        # Residual Block 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_norm1 = nn.LayerNorm(hidden_dim)
        self.res1_act = Mish()
        self.res1_drop = nn.Dropout(dropout_rate)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_norm2 = nn.LayerNorm(hidden_dim)

        # Residual Block 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res2_norm1 = nn.LayerNorm(hidden_dim // 2)
        self.res2_act = Mish()
        self.res2_drop = nn.Dropout(dropout_rate)
        self.res2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.res2_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.shortcut2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Head
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.input_layer(x)

        # Res Block 1
        res = self.res1_fc2(self.res1_drop(self.res1_act(self.res1_norm1(self.res1_fc1(h)))))
        h = nn.functional.silu(self.res1_norm2(h + res))

        # Res Block 2 with projection shortcut
        res2 = self.res2_fc2(self.res2_drop(self.res2_act(self.res2_norm1(self.res2_fc1(h)))))
        h2 = nn.functional.silu(self.res2_norm2(self.shortcut2(h) + res2))

        return self.head(h2)


class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_physics_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_age: float = 0.05,
    lambda_fc: float = 0.05
):
    """
    Composite Physics-Informed Loss:
      1. Empirical MSE loss.
      2. Monotonicity with Curing Age: dCs / dAge >= 0.
         (Compressive strength does not decrease as concrete hydrates).
      3. Monotonicity with Design Fc: dCs / dDesignFc >= 0.
         (A higher design grade concrete mix yields higher strength).
    """
    inputs.requires_grad_(True)
    preds = model(inputs)

    empirical_loss = nn.functional.mse_loss(preds, targets)

    # Compute autograd gradients of predictions w.r.t. input features
    grad_outputs = torch.ones_like(preds)
    gradients = torch.autograd.grad(
        outputs=preds,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Feature 1: Curing Age gradient
    d_cs_d_age = gradients[:, AGE_IDX]
    penalty_age = torch.relu(-d_cs_d_age).pow(2).mean()

    # Feature 0: Design Fc gradient
    d_cs_d_fc = gradients[:, DESIGN_FC_IDX]
    penalty_fc = torch.relu(-d_cs_d_fc).pow(2).mean()

    total_loss = empirical_loss + (lambda_age * penalty_age) + (lambda_fc * penalty_fc)

    return total_loss, empirical_loss.item(), (penalty_age.item() + penalty_fc.item())


def train_pinn_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LEARNING_RATE):
    """Trains the Physics-Informed Neural Network with validation checkpointing."""
    print("\n" + "=" * 80)
    print("TRAINING PHYSICS-INFORMED NEURAL NETWORK (PINN-STRENGTH)")
    print("=" * 80)

    train_dataset = ConcreteDataset(X_train, y_train)
    val_dataset = ConcreteDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConcretePINN(input_dim=4, hidden_dim=128, dropout_rate=0.15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss = float("inf")
    best_weights = None
    history = {"train_total": [], "train_mse": [], "train_physics": [], "val_mse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_phys = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss, mse_val, phys_val = compute_physics_loss(
                model, batch_X, batch_y, lambda_age=0.08, lambda_fc=0.08
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += loss.item() * len(batch_X)
            epoch_mse += mse_val * len(batch_X)
            epoch_phys += phys_val * len(batch_X)

        train_total_avg = epoch_total / len(train_dataset)
        train_mse_avg = epoch_mse / len(train_dataset)
        train_phys_avg = epoch_phys / len(train_dataset)

        # Validation evaluation
        model.eval()
        val_mse_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                val_preds = model(batch_X)
                val_mse_sum += nn.functional.mse_loss(val_preds, batch_y, reduction="sum").item()

        val_mse_avg = val_mse_sum / len(val_dataset)
        scheduler.step(val_mse_avg)

        history["train_total"].append(train_total_avg)
        history["train_mse"].append(train_mse_avg)
        history["train_physics"].append(train_phys_avg)
        history["val_mse"].append(val_mse_avg)

        if val_mse_avg < best_val_loss:
            best_val_loss = val_mse_avg
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_total_avg:.4f} (MSE: {train_mse_avg:.4f}, Phys: {train_phys_avg:.4f}) | "
                f"Val MSE: {val_mse_avg:.4f}"
            )

    model.load_state_dict(best_weights)
    print(f"PINN training completed. Best Validation MSE: {best_val_loss:.4f}")
    return model, history


# ==============================================================================
# 6. UNCERTAINTY QUANTIFICATION (MONTE CARLO DROPOUT)
# ==============================================================================

def predict_mc_dropout(model: nn.Module, X: np.ndarray, num_passes: int = 50):
    """
    Performs stochastic Monte Carlo Dropout forward passes during inference
    to estimate epistemic prediction mean and uncertainty standard deviation.
    """
    model.train()  # Keep dropout enabled during inference
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(num_passes):
            p = model(X_tensor).cpu().numpy().ravel()
            preds.append(p)

    preds = np.array(preds)  # Shape: (num_passes, N)
    mean_preds = np.mean(preds, axis=0)
    epistemic_std = np.std(preds, axis=0)
    return mean_preds, epistemic_std


# ==============================================================================
# 7. INDUCTIVE SPLIT CONFORMAL PREDICTION CALIBRATION
# ==============================================================================

class InductiveConformalCalibrator:
    """
    Split Conformal Prediction Engine providing distribution-free,
    finite-sample valid prediction intervals at guaranteed coverage level (1 - alpha).
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Computes empirical non-conformity threshold q_hat from calibration residuals."""
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)

        # Finite-sample conservative conformal quantile
        quantile_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_level = min(1.0, max(0.0, quantile_level))

        self.q_hat = float(np.quantile(residuals, quantile_level, method="higher"))
        return self.q_hat

    def predict_interval(self, y_pred: np.ndarray):
        """Constructs symmetric prediction intervals around point predictions."""
        assert self.q_hat is not None, "Calibrator must be fitted before prediction."
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper, self.q_hat

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Computes empirical test coverage and average interval width in MPa."""
        lower, upper, margin = self.predict_interval(y_pred)
        covered = (y_true >= lower) & (y_true <= upper)
        coverage_rate = np.mean(covered) * 100

        # Physical width in MPa
        width_std = upper - lower
        width_mpa = width_std * SCALER_STD_Y
        mean_width_mpa = float(np.mean(width_mpa))

        return {
            "nominal_coverage_%": (1.0 - self.alpha) * 100,
            "empirical_coverage_%": coverage_rate,
            "margin_std": margin,
            "mean_interval_width_MPa": mean_width_mpa,
        }


# ==============================================================================
# 8. VISUALIZATION SUITE
# ==============================================================================

def generate_publication_plots(
    history,
    baselines_results,
    pinn_test_pred,
    pinn_test_std,
    y_test,
    conformal_90,
    conformal_95,
    output_path="pipeline_evaluation.png"
):
    """Creates a comprehensive 4-panel diagnostic figure for the capstone report."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    y_test_mpa = to_mpa(y_test)
    pinn_pred_mpa = to_mpa(pinn_test_pred)

    # --------------------------------------------------------------------------
    # Panel 1: Parity Plot (Predicted vs Actual Strength)
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.scatter(y_test_mpa, pinn_pred_mpa, alpha=0.35, color="#1f77b4", edgecolors="none", s=25, label="Test Samples")
    ideal_line = np.linspace(y_test_mpa.min(), y_test_mpa.max(), 100)
    ax1.plot(ideal_line, ideal_line, "r--", linewidth=2.0, label="Ideal 1:1 Parity")
    ax1.set_xlabel("Measured Compressive Strength (MPa)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Compressive Strength (MPa)", fontsize=11, fontweight="bold")
    ax1.set_title("Parity Plot: Physics-Informed Neural Network", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # Panel 2: PINN Training Dynamics (Empirical vs Monotonicity Loss)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    epochs_range = range(1, len(history["train_mse"]) + 1)
    ax2.plot(epochs_range, history["train_mse"], label="Train MSE Loss", color="#2ca02c", linewidth=2.0)
    ax2.plot(epochs_range, history["val_mse"], label="Val MSE Loss", color="#d62728", linestyle="--", linewidth=2.0)
    ax2.plot(epochs_range, history["train_physics"], label="Physics Penalty (Monotonicity)", color="#9467bd", linewidth=1.5)
    ax2.set_xlabel("Training Epochs", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss Value", fontsize=11, fontweight="bold")
    ax2.set_title("Training Loss Convergence & Physics Regularization", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    # --------------------------------------------------------------------------
    # Panel 3: Conformal Prediction Intervals (Sorted by Strength)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    sort_idx = np.argsort(y_test_mpa)
    sub_indices = sort_idx[::25]  # Downsample for clear visual presentation

    sub_true = y_test_mpa[sub_indices]
    sub_pred = pinn_pred_mpa[sub_indices]

    margin_95_mpa = conformal_95.q_hat * SCALER_STD_Y
    lower_95 = sub_pred - margin_95_mpa
    upper_95 = sub_pred + margin_95_mpa

    x_coords = np.arange(len(sub_indices))
    ax3.fill_between(x_coords, lower_95, upper_95, color="#aec7e8", alpha=0.5, label="95% Conformal Interval")
    ax3.plot(x_coords, sub_pred, color="#1f77b4", linewidth=1.8, label="PINN Point Estimate")
    ax3.scatter(x_coords, sub_true, color="#d62728", s=20, zorder=5, label="Observed Ground Truth")
    ax3.set_xlabel("Ordered Test Specimens (Subset)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Compressive Strength (MPa)", fontsize=11, fontweight="bold")
    ax3.set_title("Conformal Prediction Intervals (95% Guaranteed Coverage)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # Panel 4: Model Comparison (R² & RMSE MPa)
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    all_names = list(baselines_results.keys()) + ["PINN (Proposed)"]
    pinn_test_metrics = compute_metrics(y_test, pinn_test_pred)
    all_r2 = [baselines_results[m]["test_metrics"]["R2"] for m in baselines_results] + [pinn_test_metrics["R2"]]
    all_rmse = [baselines_results[m]["test_metrics"]["RMSE_MPa"] for m in baselines_results] + [pinn_test_metrics["RMSE_MPa"]]

    x_pos = np.arange(len(all_names))
    width = 0.35

    color_r2 = "#1f77b4"
    color_rmse = "#ff7f0e"

    rects1 = ax4.bar(x_pos - width/2, all_r2, width, label="Test R² (Higher is better)", color=color_r2)
    ax4_twin = ax4.twinx()
    rects2 = ax4_twin.bar(x_pos + width/2, all_rmse, width, label="Test RMSE MPa (Lower is better)", color=color_rmse)

    ax4.set_ylabel("Test R² Score", color=color_r2, fontsize=11, fontweight="bold")
    ax4_twin.set_ylabel("Test RMSE (MPa)", color=color_rmse, fontsize=11, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_names, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax4.set_ylim([0.0, 1.05])
    ax4.set_title("Model Performance Comparison on Unseen Test Partition", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n[INFO] Diagnostic figure successfully generated and saved to: {output_path}")
    plt.close()


# ==============================================================================
# 9. COMPLETE END-TO-END EXECUTION PIPELINE
# ==============================================================================

def run_intelligent_concrete_pipeline():
    print("=" * 80)
    print("INTELLIGENT CONCRETE STRENGTH PREDICTION PIPELINE")
    print("Physics-Informed Neural Network + Baselines + Conformal Calibration")
    print("=" * 80)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_npz_data()

    # 2. Run Baselines
    baseline_results, baseline_preds = run_ml_baselines(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 3. Train Physics-Informed Neural Network
    pinn_model, history = train_pinn_model(X_train, y_train, X_val, y_val)

    # 4. Uncertainty Quantification via MC Dropout
    print("\n" + "=" * 80)
    print("UNCERTAINTY ESTIMATION (MONTE CARLO DROPOUT INFERENCE)")
    print("=" * 80)
    val_pred_pinn, val_std_pinn = predict_mc_dropout(pinn_model, X_val, num_passes=50)
    test_pred_pinn, test_std_pinn = predict_mc_dropout(pinn_model, X_test, num_passes=50)

    pinn_val_metrics = compute_metrics(y_val, val_pred_pinn)
    pinn_test_metrics = compute_metrics(y_test, test_pred_pinn)

    print(
        f"PINN Test Performance -> R²: {pinn_test_metrics['R2']:.4f} | "
        f"RMSE: {pinn_test_metrics['RMSE_MPa']:.3f} MPa | "
        f"MAE: {pinn_test_metrics['MAE_MPa']:.3f} MPa | "
        f"MAPE: {pinn_test_metrics['MAPE_%']:.2f}%"
    )
    print(
        f"Average Model Epistemic Uncertainty (Std): "
        f"{np.mean(test_std_pinn * SCALER_STD_Y):.3f} MPa"
    )

    # 5. Inductive Split Conformal Prediction Calibration
    print("\n" + "=" * 80)
    print("INDUCTIVE CONFORMAL PREDICTION CALIBRATION")
    print("=" * 80)
    calibrator_90 = InductiveConformalCalibrator(alpha=0.10)
    q90 = calibrator_90.calibrate(y_val, val_pred_pinn)
    cov_90 = calibrator_90.evaluate_coverage(y_test, test_pred_pinn)

    calibrator_95 = InductiveConformalCalibrator(alpha=0.05)
    q95 = calibrator_95.calibrate(y_val, val_pred_pinn)
    cov_95 = calibrator_95.evaluate_coverage(y_test, test_pred_pinn)

    print(
        f"Conformal 90% -> Empirical Coverage: {cov_90['empirical_coverage_%']:.2f}% | "
        f"Margin: ±{cov_90['margin_std'] * SCALER_STD_Y:.3f} MPa | "
        f"Mean Interval Width: {cov_90['mean_interval_width_MPa']:.3f} MPa"
    )
    print(
        f"Conformal 95% -> Empirical Coverage: {cov_95['empirical_coverage_%']:.2f}% | "
        f"Margin: ±{cov_95['margin_std'] * SCALER_STD_Y:.3f} MPa | "
        f"Mean Interval Width: {cov_95['mean_interval_width_MPa']:.3f} MPa"
    )

    # 6. Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (TEST SPLIT)")
    print("=" * 80)

    summary_rows = []
    for name, res in baseline_results.items():
        m = res["test_metrics"]
        summary_rows.append({
            "Model": name,
            "R² Score": f"{m['R2']:.4f}",
            "RMSE (MPa)": f"{m['RMSE_MPa']:.3f}",
            "MAE (MPa)": f"{m['MAE_MPa']:.3f}",
            "MAPE (%)": f"{m['MAPE_%']:.2f}%"
        })
    summary_rows.append({
        "Model": "PINN (Proposed)",
        "R² Score": f"{pinn_test_metrics['R2']:.4f}",
        "RMSE (MPa)": f"{pinn_test_metrics['RMSE_MPa']:.3f}",
        "MAE (MPa)": f"{pinn_test_metrics['MAE_MPa']:.3f}",
        "MAPE (%)": f"{pinn_test_metrics['MAPE_%']:.2f}%"
    })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # 7. Publication Diagnostic Plots
    generate_publication_plots(
        history=history,
        baselines_results=baseline_results,
        pinn_test_pred=test_pred_pinn,
        pinn_test_std=test_std_pinn,
        y_test=y_test,
        conformal_90=calibrator_90,
        conformal_95=calibrator_95,
        output_path="pipeline_evaluation.png"
    )

    print("\n[SUCCESS] Intelligent Concrete Strength Prediction Pipeline completed successfully.")
    return {
        "baseline_results": baseline_results,
        "pinn_model": pinn_model,
        "pinn_test_metrics": pinn_test_metrics,
        "conformal_90": cov_90,
        "conformal_95": cov_95,
    }


if __name__ == "__main__":
    run_intelligent_concrete_pipeline()
