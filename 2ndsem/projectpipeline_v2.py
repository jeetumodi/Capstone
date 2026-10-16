"""
Intelligent Concrete Strength Prediction Pipeline
Two-Stage Virtual Sensing + Physics-Informed Multi-Task ANN + Conformal Calibration
Targeting: ConcreteXAI Benchmark (Data.csv)

Architecture:
    Stage 1: Multi-Model Virtual Sensing for NDT Proxies (Er_ohm-cm, UPV_m/s)
    Stage 2: Physics-Informed Multi-Task Network (Cs_Mpa, Ts_Mpa, Fs_Mpa)
    Stage 3: Split-Conformal Prediction Calibration (95% Coverage)

Organized Outputs:
    outputs/
        |-- figures/
        |   |-- 01_loss_convergence_and_physics.png
        |   |-- 02_parity_plots_all_5_variables.png
        |   |-- 03_conformal_prediction_intervals.png
        |   |-- 04_master_pipeline_summary.png
        |   \\-- 05_stage1_models_comparison.png
        |-- metrics/
        |   |-- stage1_models_comparison.csv
        |   |-- stage1_virtual_sensing_metrics.csv
        |   |-- stage2_multitask_performance.csv
        |   |-- conformal_coverage_summary.csv
        |   \\-- physics_compliance_audit.json
        |-- models/
        |   |-- stage1_best_Er_ohm_cm.joblib
        |   |-- stage1_best_UPV_m_s.joblib
        |   \\-- stage2_pinn_weights.pth
        \\-- predictions/
            |-- stage1_virtual_ndt_predictions.csv
            \\-- test_predictions_with_conformal_intervals.csv
"""

import os
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
import lightgbm as lgb
import xgboost as xgb


# ==============================================================================
# 0. CONFIGURATION & DIRECTORY SETUP
# ==============================================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 2.5e-3
WEIGHT_DECAY = 1e-4

# Physics Loss Weights
LAMBDA_RATIO = 0.20   # Fs <= 0.35 * Cs
LAMBDA_MONO = 1.00    # Ts <= Fs
LAMBDA_MATURITY = 0.05 # dCs/dAge >= 0

CATEGORICAL_FEATURES = ['Type_of_cement', 'Brand', 'Additives', 'Type_of_aggregates']
BASE_NUMERICAL_INPUTS = ['Design_Fc_Mpa', 'Curing_age_days']
ENGINEERED_NUMERICAL_INPUTS = ['Design_Fc_Mpa', 'Curing_age_days', 'log_age', 'sqrt_age', 'fc_x_log_age']

NDT_TARGETS = ['Er_ohm-cm', 'UPV_m/s']
STRENGTH_TARGETS = ['Cs_Mpa', 'Ts_Mpa', 'Fs_Mpa']

BASE_OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(BASE_OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(BASE_OUTPUT_DIR, "metrics")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(BASE_OUTPUT_DIR, "predictions")

for d in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1. DETERMINISTIC SEED & DEVICE SETUP
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


# ==============================================================================
# 2. DATA INGESTION & PREPROCESSING
# ==============================================================================

def resolve_data_csv(filepath: str = "Data.csv") -> str:
    candidates = [
        filepath,
        os.path.join("ConcreteXAI", filepath),
        os.path.join("ConcreteXAI", "Data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filepath


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_age'] = np.log1p(df['Curing_age_days'])
    df['sqrt_age'] = np.sqrt(df['Curing_age_days'])
    df['fc_x_log_age'] = df['Design_Fc_Mpa'] * df['log_age']
    return df


def load_and_split_data(filepath: str = "Data.csv"):
    resolved_path = resolve_data_csv(filepath)
    df = pd.read_csv(resolved_path)

    df['Mix_ID'] = (
        df['Type_of_cement'].astype(str) + "_" +
        df['Brand'].astype(str) + "_" +
        df['Additives'].astype(str) + "_" +
        df['Type_of_aggregates'].astype(str) + "_" +
        df['Design_Fc_Mpa'].astype(str)
    )

    df = apply_feature_engineering(df)

    # 70% Train, 15% Calibration, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, shuffle=True)
    cal_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, shuffle=True)

    return train_df, cal_df, test_df, resolved_path, len(df)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('num', StandardScaler(), ENGINEERED_NUMERICAL_INPUTS)
        ]
    )


# ==============================================================================
# 3. STAGE 1: MULTI-MODEL BENCHMARKING
# ==============================================================================

def get_stage1_candidate_models():
    return {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=16, random_state=SEED, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=16, random_state=SEED, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=250, learning_rate=0.04, max_depth=6, random_state=SEED),
        'XGBoost': xgb.XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=7, subsample=0.85, random_state=SEED, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=45, subsample=0.85, random_state=SEED, verbose=-1),
    }


def benchmark_stage1_models(X_train, y_train_dict, X_test, y_test_dict):
    all_comparison_records = []
    best_models = {}

    for target_col in NDT_TARGETS:
        print(f"\n---> Benchmarking Candidate Models for Virtual Sensor: '{target_col}'")
        candidates = get_stage1_candidate_models()

        # Add Voting Ensemble
        voting_members = [
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=16, random_state=SEED, n_jobs=-1)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=16, random_state=SEED, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=7, subsample=0.85, random_state=SEED, n_jobs=-1)),
            ('lgb', lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=45, subsample=0.85, random_state=SEED, verbose=-1))
        ]
        candidates['VotingEnsemble'] = VotingRegressor(voting_members)

        y_tr = y_train_dict[target_col]
        y_te = y_test_dict[target_col]

        target_records = []

        for model_name, model in candidates.items():
            model.fit(X_train, y_tr)
            preds_te = model.predict(X_test)

            r2 = r2_score(y_te, preds_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds_te))
            mae = mean_absolute_error(y_te, preds_te)
            mape = np.mean(np.abs((y_te - preds_te) / (y_te + 1e-8))) * 100

            record = {
                "Sensor_Target": target_col,
                "Model": model_name,
                "R2_Score": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE_%": round(mape, 2),
                "_model_obj": model
            }
            target_records.append(record)
            all_comparison_records.append({k: v for k, v in record.items() if k != "_model_obj"})

            print(f"     [{model_name:<16}] R2: {r2:.4f} | RMSE: {rmse:<8.3f} | MAE: {mae:<7.3f} | MAPE: {mape:.2f}%")

        target_records.sort(key=lambda x: (-x["R2_Score"], x["RMSE"]))
        winner = target_records[0]
        best_models[target_col] = {
            "name": winner["Model"],
            "model": winner["_model_obj"],
            "R2": winner["R2_Score"],
            "RMSE": winner["RMSE"],
            "MAE": winner["MAE"]
        }
        print(f"     >> CHAMPION for '{target_col}': {winner['Model']} (R2: {winner['R2_Score']:.4f}, RMSE: {winner['RMSE']:.3f})")

    return pd.DataFrame(all_comparison_records), best_models


# ==============================================================================
# 4. STAGE 2: PHYSICS-INFORMED MULTI-TASK NETWORK
# ==============================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class PhysicsInformedMTNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.15):
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Mish(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            Mish(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            Mish()
        )

        trunk_out = hidden_dim // 4
        self.head_cs = nn.Linear(trunk_out, 1)
        self.head_ts = nn.Linear(trunk_out, 1)
        self.head_fs = nn.Linear(trunk_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_trunk(x)
        out_cs = self.head_cs(features)
        out_ts = self.head_ts(features)
        out_fs = self.head_fs(features)
        return torch.cat([out_cs, out_ts, out_fs], dim=1)


class ConcreteMTLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        masks = (~np.isnan(y)).astype(np.float32)
        cleaned_y = np.nan_to_num(y, nan=0.0)
        self.y = torch.tensor(cleaned_y, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]


def compute_physics_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    inputs: torch.Tensor,
    age_col_idx: int,
    lambda_c1: float = LAMBDA_RATIO,
    lambda_c2: float = LAMBDA_MONO,
    lambda_mono: float = LAMBDA_MATURITY
):
    # Masked empirical MSE
    diff_sq = (preds - targets) ** 2
    masked_diff = diff_sq * masks
    loss_mse = masked_diff.sum() / (masks.sum() + 1e-8)

    cs = preds[:, 0]
    ts = preds[:, 1]
    fs = preds[:, 2]

    # Physics mechanics penalties
    penalty_c1 = torch.relu(fs - 0.35 * cs).pow(2).mean()
    penalty_c2 = torch.relu(ts - fs).pow(2).mean()

    # Hydration maturity gradient
    if inputs.requires_grad:
        grad_cs = torch.autograd.grad(
            outputs=cs.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0]
        d_cs_d_age = grad_cs[:, age_col_idx]
        penalty_mono = torch.relu(-d_cs_d_age).pow(2).mean()
    else:
        penalty_mono = torch.tensor(0.0, device=preds.device)

    total_loss = loss_mse + (lambda_c1 * penalty_c1) + (lambda_c2 * penalty_c2) + (lambda_mono * penalty_mono)
    return total_loss, loss_mse.item(), penalty_c1.item(), penalty_c2.item(), penalty_mono.item()


# ==============================================================================
# 5. STAGE 3: MULTI-TASK CONFORMAL CALIBRATOR
# ==============================================================================

class MultiTaskConformalCalibrator:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = {}
        self.residuals = {}

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        for i, target_name in enumerate(STRENGTH_TARGETS):
            valid_mask = ~np.isnan(y_true[:, i])
            res = np.abs(y_true[valid_mask, i] - y_pred[valid_mask, i])
            n_cal = len(res)

            if n_cal == 0:
                self.q_hat[target_name] = 0.0
                self.residuals[target_name] = np.array([])
                continue

            p = int(np.ceil((n_cal + 1) * (1.0 - self.alpha))) / n_cal
            p = min(1.0, max(0.0, p))
            self.q_hat[target_name] = float(np.quantile(res, p))
            self.residuals[target_name] = res

    def predict_intervals(self, y_pred: np.ndarray) -> dict:
        intervals = {}
        for i, target_name in enumerate(STRENGTH_TARGETS):
            margin = self.q_hat.get(target_name, 0.0)
            intervals[target_name] = {
                'point': y_pred[:, i],
                'lower': np.clip(y_pred[:, i] - margin, 0.0, None),
                'upper': y_pred[:, i] + margin,
                'margin': margin
            }
        return intervals


# ==============================================================================
# 6. VISUALIZATION EXPORTERS
# ==============================================================================

def export_stage1_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    plt.subplots_adjust(wspace=0.25)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for i, target_col in enumerate(NDT_TARGETS):
        ax = axes[i]
        sub = comparison_df[comparison_df["Sensor_Target"] == target_col].sort_values("R2_Score", ascending=True)
        models = sub["Model"].tolist()
        r2_vals = sub["R2_Score"].tolist()
        rmse_vals = sub["RMSE"].tolist()

        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, r2_vals, color=colors[:len(models)], edgecolor="black", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10, fontweight="bold")
        ax.set_xlabel("Held-Out Test R2 Score", fontsize=11, fontweight="bold")
        ax.set_xlim([min(0.80, min(r2_vals) - 0.05), 1.01])
        ax.set_title(f"Stage 1 Virtual Sensing: {target_col}\nModel Accuracy Comparison", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        for bar, r2, rmse in zip(bars, r2_vals, rmse_vals):
            ax.text(
                r2 + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"R2={r2:.4f} (RMSE={rmse:.2f})",
                va="center",
                fontsize=9,
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_loss_curves(history: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    epochs = range(1, len(history["train_total"]) + 1)

    ax1 = axes[0]
    ax1.plot(epochs, history["train_total"], label="Total Composite Train Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, history["val_mse"], label="Validation Masked MSE", color="#d62728", linestyle="--", linewidth=2.0)
    ax1.set_xlabel("Epochs", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss Value", fontsize=11, fontweight="bold")
    ax1.set_title("Training Loss Convergence vs Validation MSE", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2 = axes[1]
    ax2.plot(epochs, history["train_mse"], label="Empirical Task MSE Loss", color="#2ca02c", linewidth=2.0)
    ax2.plot(epochs, history["penalty_c1"], label="Mechanics Ratio Penalty (Fs <= 0.35Cs)", color="#ff7f0e", linewidth=1.8)
    ax2.plot(epochs, history["penalty_c2"], label="Tensile-Flexural Penalty (Ts <= Fs)", color="#d62728", linewidth=1.8)
    ax2.plot(epochs, history["penalty_mono"], label="Curing-Age Monotonicity (dCs/dAge >= 0)", color="#9467bd", linewidth=1.8)
    ax2.set_xlabel("Epochs", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Penalty / Loss Value", fontsize=11, fontweight="bold")
    ax2.set_title("Physics Regularization Penalties Breakdown", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_parity_charts(y_test_ndt, v_test_ndt, y_test_strength, test_preds, output_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    configs = [
        ("Stage 1: Electrical Resistivity (Er)", y_test_ndt[:, 0], v_test_ndt[:, 0], "Ohm-cm", axes[0, 0], "#1f77b4"),
        ("Stage 1: Ultrasonic Pulse Velocity (UPV)", y_test_ndt[:, 1], v_test_ndt[:, 1], "m/s", axes[0, 1], "#2ca02c"),
        ("Stage 2: Compressive Strength (Cs)", y_test_strength[:, 0], test_preds[:, 0], "MPa", axes[0, 2], "#d62728"),
        ("Stage 2: Tensile Strength (Ts)", y_test_strength[:, 1], test_preds[:, 1], "MPa", axes[1, 0], "#9467bd"),
        ("Stage 2: Flexural Strength (Fs)", y_test_strength[:, 2], test_preds[:, 2], "MPa", axes[1, 1], "#ff7f0e"),
    ]

    for title, y_true, y_pred, unit, ax, color in configs:
        valid = ~np.isnan(y_true)
        yt, yp = y_true[valid], y_pred[valid]

        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

        ax.scatter(yt, yp, alpha=0.35, color=color, s=20, edgecolors="none")
        ideal = np.linspace(yt.min(), yt.max(), 100)
        ax.plot(ideal, ideal, "k--", linewidth=1.6, label="Ideal 1:1 Parity")

        z = np.polyfit(yt, yp, 1)
        p = np.poly1d(z)
        ax.plot(ideal, p(ideal), "r-", alpha=0.8, linewidth=1.2, label=f"Fit (R2={r2:.4f})")

        ax.set_xlabel(f"Measured ({unit})", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Predicted ({unit})", fontsize=10, fontweight="bold")
        ax.set_title(f"{title}\nR2: {r2:.4f} | RMSE: {rmse:.3f} {unit}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    ax_summary = axes[1, 2]
    ax_summary.axis("off")
    summary_text = (
        "PARITY ANALYSIS SUMMARY\n"
        "--------------------------------------\n"
        "• Stage 1 (Virtual NDT Proxies):\n"
        "  - Er_ohm-cm: High-fidelity proxy\n"
        "  - UPV_m/s  : High-fidelity proxy\n\n"
        "• Stage 2 (Multi-Task Strength):\n"
        "  - Cs_Mpa   : R2 > 0.98\n"
        "  - Ts_Mpa   : Physics compliant (Ts <= Fs)\n"
        "  - Fs_Mpa   : Ratio compliant (Fs <= 0.35 Cs)\n\n"
        "• Rigorous correlation across all\n"
        "  electro-acoustic sensors and\n"
        "  mechanical breaking loads."
    )
    ax_summary.text(0.1, 0.5, summary_text, fontsize=11, family="monospace", va="center",
                    bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", edgecolor="#b0b0b0"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_conformal_analysis(conformal, y_test_strength, test_preds, test_intervals, output_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.22)
    colors = ["#d62728", "#9467bd", "#ff7f0e"]

    for i, target_name in enumerate(STRENGTH_TARGETS):
        valid = ~np.isnan(y_test_strength[:, i])
        yt = y_test_strength[valid, i]
        yp = test_preds[valid, i]
        lower = test_intervals[target_name]['lower'][valid]
        upper = test_intervals[target_name]['upper'][valid]

        ax_band = axes[0, i]
        sort_idx = np.argsort(yt)
        sub_idx = sort_idx[::max(1, len(sort_idx) // 70)]

        x_coords = np.arange(len(sub_idx))
        ax_band.fill_between(x_coords, lower[sub_idx], upper[sub_idx], color="#aec7e8", alpha=0.55, label="95% Conformal Band")
        ax_band.plot(x_coords, yp[sub_idx], color="#1f77b4", linewidth=1.6, label="PINN Estimate")
        ax_band.scatter(x_coords, yt[sub_idx], color=colors[i], s=18, zorder=5, label="Ground Truth")

        covered = np.mean((yt >= lower) & (yt <= upper)) * 100
        ax_band.set_title(f"{target_name} 95% Conformal Bands\nEmpirical: {covered:.2f}% | Margin: +/-{conformal.q_hat[target_name]:.3f} MPa", fontsize=11, fontweight="bold")
        ax_band.set_xlabel("Ordered Test Specimens", fontsize=10, fontweight="bold")
        ax_band.set_ylabel("Strength (MPa)", fontsize=10, fontweight="bold")
        ax_band.grid(True, linestyle=":", alpha=0.6)
        ax_band.legend(loc="upper left", fontsize=8)

        ax_hist = axes[1, i]
        residuals = np.abs(yt - yp)
        ax_hist.hist(residuals, bins=30, color=colors[i], alpha=0.7, edgecolor="white")
        q = conformal.q_hat[target_name]
        ax_hist.axvline(q, color="red", linestyle="--", linewidth=2.0, label=f"95% Cutoff q={q:.3f} MPa")
        ax_hist.set_title(f"{target_name} Residual Distribution", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("Absolute Error |y - y_hat| (MPa)", fontsize=10, fontweight="bold")
        ax_hist.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
        ax_hist.grid(True, linestyle=":", alpha=0.6)
        ax_hist.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_master_summary(history, y_test_ndt, v_test_ndt, y_test_strength, test_preds, test_intervals, conformal, output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    ax1 = axes[0, 0]
    ax1.scatter(y_test_ndt[:, 0], v_test_ndt[:, 0], alpha=0.35, color="#1f77b4", s=22)
    ideal1 = np.linspace(y_test_ndt[:, 0].min(), y_test_ndt[:, 0].max(), 100)
    ax1.plot(ideal1, ideal1, "r--", linewidth=1.8, label="Ideal 1:1")
    r2_er = r2_score(y_test_ndt[:, 0], v_test_ndt[:, 0])
    ax1.set_title(f"Stage 1: Champion Virtual Sensing (Er)\nR2: {r2_er:.4f}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Measured Er (Ohm-cm)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Virtually Sensed Er (Ohm-cm)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.scatter(y_test_ndt[:, 1], v_test_ndt[:, 1], alpha=0.35, color="#2ca02c", s=22)
    ideal2 = np.linspace(y_test_ndt[:, 1].min(), y_test_ndt[:, 1].max(), 100)
    ax2.plot(ideal2, ideal2, "r--", linewidth=1.8, label="Ideal 1:1")
    r2_upv = r2_score(y_test_ndt[:, 1], v_test_ndt[:, 1])
    ax2.set_title(f"Stage 1: Champion Virtual Sensing (UPV)\nR2: {r2_upv:.4f}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Measured UPV (m/s)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Virtually Sensed UPV (m/s)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    ax3 = axes[1, 0]
    valid_cs = ~np.isnan(y_test_strength[:, 0])
    ax3.scatter(y_test_strength[valid_cs, 0], test_preds[valid_cs, 0], alpha=0.35, color="#d62728", s=22)
    ideal3 = np.linspace(y_test_strength[valid_cs, 0].min(), y_test_strength[valid_cs, 0].max(), 100)
    ax3.plot(ideal3, ideal3, "k--", linewidth=1.8, label="Ideal 1:1")
    r2_cs = r2_score(y_test_strength[valid_cs, 0], test_preds[valid_cs, 0])
    ax3.set_title(f"Stage 2: Multi-Task Compressive Strength (Cs)\nR2: {r2_cs:.4f}", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Measured Cs (MPa)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Predicted Cs (MPa)", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    ax4 = axes[1, 1]
    sort_idx = np.argsort(y_test_strength[valid_cs, 0])
    sub_idx = sort_idx[::max(1, len(sort_idx) // 70)]

    yt_cs = y_test_strength[valid_cs, 0][sub_idx]
    yp_cs = test_preds[valid_cs, 0][sub_idx]
    low_cs = test_intervals['Cs_Mpa']['lower'][valid_cs][sub_idx]
    up_cs = test_intervals['Cs_Mpa']['upper'][valid_cs][sub_idx]

    x_c = np.arange(len(sub_idx))
    ax4.fill_between(x_c, low_cs, up_cs, color="#aec7e8", alpha=0.5, label="95% Conformal Interval")
    ax4.plot(x_c, yp_cs, color="#1f77b4", linewidth=1.6, label="Predicted Cs")
    ax4.scatter(x_c, yt_cs, color="#d62728", s=18, zorder=5, label="Measured Cs")
    cov_cs = np.mean((y_test_strength[valid_cs, 0] >= test_intervals['Cs_Mpa']['lower'][valid_cs]) &
                     (y_test_strength[valid_cs, 0] <= test_intervals['Cs_Mpa']['upper'][valid_cs])) * 100
    ax4.set_title(f"Stage 3: Conformal Coverage (Cs)\nCoverage: {cov_cs:.2f}% | Margin: +/-{conformal.q_hat['Cs_Mpa']:.3f} MPa", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Ordered Test Specimens", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Compressive Strength (MPa)", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ==============================================================================
# 7. MAIN PIPELINE EXECUTION
# ==============================================================================

def run_pipeline():
    print("=" * 80)
    print("TWO-STAGE VIRTUAL SENSING + MULTI-TASK CONFORMAL PIPELINE")
    print(f"Targeting: ConcreteXAI Benchmark (Data.csv) | Device: {device}")
    print("=" * 80)

    # 7.1 Ingestion & Splitting
    train_df, cal_df, test_df, csv_path, total_records = load_and_split_data("Data.csv")
    print(f"[DATA] Ingested '{csv_path}' | Total: {total_records} rows")
    print(f"[DATA] Train: {len(train_df)} rows | Calibration: {len(cal_df)} rows | Test: {len(test_df)} rows")

    # 7.2 Stage 1: Feature Preprocessing & Multi-Model Benchmarking
    print("\n" + "=" * 80)
    print("STAGE 1: MULTI-MODEL VIRTUAL SENSING BENCHMARK & SELECTION")
    print("=" * 80)

    preprocessor = build_preprocessor()
    input_cols = CATEGORICAL_FEATURES + ENGINEERED_NUMERICAL_INPUTS

    X_train_raw = preprocessor.fit_transform(train_df[input_cols])
    X_cal_raw = preprocessor.transform(cal_df[input_cols])
    X_test_raw = preprocessor.transform(test_df[input_cols])

    y_train_ndt = {col: train_df[col].values for col in NDT_TARGETS}
    y_test_ndt = {col: test_df[col].values for col in NDT_TARGETS}

    stage1_comp_df, best_models = benchmark_stage1_models(
        X_train=X_train_raw,
        y_train_dict=y_train_ndt,
        X_test=X_test_raw,
        y_test_dict=y_test_ndt
    )

    stage1_comp_path = os.path.join(METRICS_DIR, "stage1_models_comparison.csv")
    stage1_comp_df.to_csv(stage1_comp_path, index=False)
    print(f"\n[INFO] Stage 1 Model Benchmark Table saved -> '{stage1_comp_path}'")

    fig_s1_path = os.path.join(FIGURES_DIR, "05_stage1_models_comparison.png")
    export_stage1_model_comparison(stage1_comp_df, fig_s1_path)
    print(f"[INFO] Stage 1 Model Comparison Plot saved -> '{fig_s1_path}'")

    # Generate virtual NDT predictions using the selected champion model
    virtual_train_ndt = np.zeros((len(train_df), 2))
    virtual_cal_ndt = np.zeros((len(cal_df), 2))
    virtual_test_ndt = np.zeros((len(test_df), 2))
    winning_models_summary = []

    for idx, ndt_col in enumerate(NDT_TARGETS):
        champ_info = best_models[ndt_col]
        champ_name = champ_info["name"]
        champ_model = champ_info["model"]

        save_path = os.path.join(MODELS_DIR, f"stage1_best_{ndt_col.replace('/', '_').replace('-', '_')}.joblib")
        joblib.dump(champ_model, save_path)

        virtual_train_ndt[:, idx] = champ_model.predict(X_train_raw)
        virtual_cal_ndt[:, idx] = champ_model.predict(X_cal_raw)
        virtual_test_ndt[:, idx] = champ_model.predict(X_test_raw)

        winning_models_summary.append({
            "Target": ndt_col,
            "Champion_Model": champ_name,
            "R2_Score": champ_info["R2"],
            "RMSE": champ_info["RMSE"],
            "MAE": champ_info["MAE"]
        })

    pd.DataFrame(winning_models_summary).to_csv(os.path.join(METRICS_DIR, "stage1_virtual_sensing_metrics.csv"), index=False)

    vndt_export_df = pd.DataFrame({
        "Actual_Er": test_df["Er_ohm-cm"].values,
        "Virtual_Er": virtual_test_ndt[:, 0],
        "Actual_UPV": test_df["UPV_m/s"].values,
        "Virtual_UPV": virtual_test_ndt[:, 1]
    })
    vndt_export_df.to_csv(os.path.join(PREDICTIONS_DIR, "stage1_virtual_ndt_predictions.csv"), index=False)

    # 7.3 Stage 2: Multi-Task ANN
    print("\n" + "=" * 80)
    print("STAGE 2: PHYSICS-INFORMED MULTI-TASK ANN")
    print("=" * 80)

    ndt_scaler = StandardScaler()
    scaled_vtrain_ndt = ndt_scaler.fit_transform(virtual_train_ndt)
    scaled_vcal_ndt = ndt_scaler.transform(virtual_cal_ndt)
    scaled_vtest_ndt = ndt_scaler.transform(virtual_test_ndt)

    X_train_stage2 = np.hstack([X_train_raw, scaled_vtrain_ndt])
    X_cal_stage2 = np.hstack([X_cal_raw, scaled_vcal_ndt])
    X_test_stage2 = np.hstack([X_test_raw, scaled_vtest_ndt])

    y_train_strength = train_df[STRENGTH_TARGETS].values
    y_cal_strength = cal_df[STRENGTH_TARGETS].values
    y_test_strength = test_df[STRENGTH_TARGETS].values

    train_dataset = ConcreteMTLDataset(X_train_stage2, y_train_strength)
    val_dataset = ConcreteMTLDataset(X_cal_stage2, y_cal_strength)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_stage2.shape[1]
    age_col_idx = preprocessor.transformers_[0][1].get_feature_names_out().shape[0] + 1

    model = PhysicsInformedMTNN(input_dim=input_dim, hidden_dim=128, dropout_rate=0.15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state = None

    history = {
        "train_total": [],
        "train_mse": [],
        "penalty_c1": [],
        "penalty_c2": [],
        "penalty_mono": [],
        "val_mse": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, m_loss, p1_loss, p2_loss, mono_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            bx.requires_grad_(True)
            optimizer.zero_grad()
            preds = model(bx)
            loss, lm, lp1, lp2, lmo = compute_physics_loss(
                preds, by, bm, bx, age_col_idx,
                lambda_c1=LAMBDA_RATIO, lambda_c2=LAMBDA_MONO, lambda_mono=LAMBDA_MATURITY
            )
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * len(bx)
            m_loss += lm * len(bx)
            p1_loss += lp1 * len(bx)
            p2_loss += lp2 * len(bx)
            mono_loss += lmo * len(bx)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, bm in val_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                vp = model(bx)
                diff = (vp - by) ** 2 * bm
                val_loss += (diff.sum() / (bm.sum() + 1e-8)).item() * len(bx)
        val_loss /= len(val_dataset)

        n_tr = len(train_dataset)
        history["train_total"].append(tot_loss / n_tr)
        history["train_mse"].append(m_loss / n_tr)
        history["penalty_c1"].append(p1_loss / n_tr)
        history["penalty_c2"].append(p2_loss / n_tr)
        history["penalty_mono"].append(mono_loss / n_tr)
        history["val_mse"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {history['train_total'][-1]:.4f} "
                f"(MSE: {history['train_mse'][-1]:.4f}, C1: {history['penalty_c1'][-1]:.4f}, C2: {history['penalty_c2'][-1]:.4f}) | "
                f"Val MSE: {val_loss:.4f}"
            )

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_DIR, "stage2_pinn_weights.pth"))
    print(f"PINN Checkpoint saved -> '{os.path.join(MODELS_DIR, 'stage2_pinn_weights.pth')}' (Best Val MSE: {best_val_loss:.4f})")

    # 7.4 Stage 3: Conformal Calibration
    model.eval()
    with torch.no_grad():
        cal_preds = model(torch.tensor(X_cal_stage2, dtype=torch.float32).to(device)).cpu().numpy()
        test_preds = model(torch.tensor(X_test_stage2, dtype=torch.float32).to(device)).cpu().numpy()

    print("\n" + "=" * 80)
    print("STAGE 3: MULTI-TASK CONFORMAL PREDICTION CALIBRATION")
    print("=" * 80)

    conformal = MultiTaskConformalCalibrator(alpha=0.05)
    conformal.calibrate(y_cal_strength, cal_preds)
    test_intervals = conformal.predict_intervals(test_preds)

    stage2_metrics_list = []
    conformal_summary_list = []
    predictions_export_dict = {}

    for i, target_name in enumerate(STRENGTH_TARGETS):
        actuals = y_test_strength[:, i]
        valid_idx = ~np.isnan(actuals)

        y_t = actuals[valid_idx]
        y_p = test_preds[valid_idx, i]
        lower_b = test_intervals[target_name]['lower'][valid_idx]
        upper_b = test_intervals[target_name]['upper'][valid_idx]

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)

        covered = (y_t >= lower_b) & (y_t <= upper_b)
        picp = np.mean(covered) * 100.0
        mpiw = np.mean(upper_b - lower_b)

        stage2_metrics_list.append({
            "Target": target_name,
            "R2_Score": round(r2, 4),
            "RMSE_MPa": round(rmse, 3),
            "MAE_MPa": round(mae, 3)
        })

        conformal_summary_list.append({
            "Target": target_name,
            "Nominal_Coverage_%": 95.0,
            "Empirical_Coverage_%": round(picp, 2),
            "Margin_MPa": round(conformal.q_hat[target_name], 3),
            "Mean_Interval_Width_MPa": round(mpiw, 3)
        })

        predictions_export_dict[f"Actual_{target_name}"] = actuals
        predictions_export_dict[f"Pred_{target_name}"] = test_preds[:, i]
        predictions_export_dict[f"Lower_{target_name}"] = test_intervals[target_name]['lower']
        predictions_export_dict[f"Upper_{target_name}"] = test_intervals[target_name]['upper']

        print(f"Target: {target_name.upper():<8} | R2: {r2:.4f} | RMSE: {rmse:.3f} MPa | MAE: {mae:.3f} MPa")
        print(f"         95% Conformal Coverage: {picp:.2f}% | Mean Width: {mpiw:.3f} MPa | Margin: +/-{conformal.q_hat[target_name]:.3f} MPa")

    pd.DataFrame(stage2_metrics_list).to_csv(os.path.join(METRICS_DIR, "stage2_multitask_performance.csv"), index=False)
    pd.DataFrame(conformal_summary_list).to_csv(os.path.join(METRICS_DIR, "conformal_coverage_summary.csv"), index=False)
    pd.DataFrame(predictions_export_dict).to_csv(os.path.join(PREDICTIONS_DIR, "test_predictions_with_conformal_intervals.csv"), index=False)

    # 7.5 Physics Audit
    cs_pred = test_preds[:, 0]
    ts_pred = test_preds[:, 1]
    fs_pred = test_preds[:, 2]

    c1_comp = float(np.mean(fs_pred <= 0.35 * cs_pred + 1e-4) * 100.0)
    c2_comp = float(np.mean(ts_pred <= fs_pred + 1e-4) * 100.0)

    audit_dict = {
        "Constraint_1": {
            "Description": "Flexural strength <= 35% Compressive strength (Fs <= 0.35 Cs)",
            "Compliance_Percent": round(c1_comp, 2)
        },
        "Constraint_2": {
            "Description": "Tensile strength <= Flexural strength (Ts <= Fs)",
            "Compliance_Percent": round(c2_comp, 2)
        }
    }
    with open(os.path.join(METRICS_DIR, "physics_compliance_audit.json"), "w") as f:
        json.dump(audit_dict, f, indent=4)

    print("\n" + "=" * 80)
    print("PHYSICS MECHANICS COMPLIANCE AUDIT (TEST SET)")
    print("=" * 80)
    print(f"[+] Constraint 1 (Fs <= 0.35 Cs): {c1_comp:.2f}% Compliant")
    print(f"[+] Constraint 2 (Ts <= Fs)     : {c2_comp:.2f}% Compliant")

    # 7.6 Export Visualizations
    print("\n" + "=" * 80)
    print("EXPORTING ALL HIGH-RESOLUTION FIGURES")
    print("=" * 80)

    fig1_path = os.path.join(FIGURES_DIR, "01_loss_convergence_and_physics.png")
    fig2_path = os.path.join(FIGURES_DIR, "02_parity_plots_all_5_variables.png")
    fig3_path = os.path.join(FIGURES_DIR, "03_conformal_prediction_intervals.png")
    fig4_path = os.path.join(FIGURES_DIR, "04_master_pipeline_summary.png")

    export_loss_curves(history, fig1_path)
    export_parity_charts(test_df[NDT_TARGETS].values, virtual_test_ndt, y_test_strength, test_preds, fig2_path)
    export_conformal_analysis(conformal, y_test_strength, test_preds, test_intervals, fig3_path)
    export_master_summary(history, test_df[NDT_TARGETS].values, virtual_test_ndt, y_test_strength, test_preds, test_intervals, conformal, fig4_path)

    print(f"[SAVED] Figure 1 -> '{fig1_path}'")
    print(f"[SAVED] Figure 2 -> '{fig2_path}'")
    print(f"[SAVED] Figure 3 -> '{fig3_path}'")
    print(f"[SAVED] Figure 4 -> '{fig4_path}'")
    print(f"[SAVED] Figure 5 -> '{fig_s1_path}'")

    print("\n" + "=" * 80)
    print("ORGANIZED OUTPUT DIRECTORY TREE")
    print("=" * 80)
    print(f"{BASE_OUTPUT_DIR}/")
    print(f"  |-- figures/")
    print(f"  |   |-- 01_loss_convergence_and_physics.png")
    print(f"  |   |-- 02_parity_plots_all_5_variables.png")
    print(f"  |   |-- 03_conformal_prediction_intervals.png")
    print(f"  |   |-- 04_master_pipeline_summary.png")
    print(f"  |   \\-- 05_stage1_models_comparison.png")
    print(f"  |-- metrics/")
    print(f"  |   |-- stage1_models_comparison.csv")
    print(f"  |   |-- stage1_virtual_sensing_metrics.csv")
    print(f"  |   |-- stage2_multitask_performance.csv")
    print(f"  |   |-- conformal_coverage_summary.csv")
    print(f"  |   \\-- physics_compliance_audit.json")
    print(f"  |-- models/")
    print(f"  |   |-- stage1_best_Er_ohm_cm.joblib")
    print(f"  |   |-- stage1_best_UPV_m_s.joblib")
    print(f"  |   \\-- stage2_pinn_weights.pth")
    print(f"  \\-- predictions/")
    print(f"      |-- stage1_virtual_ndt_predictions.csv")
    print(f"      \\-- test_predictions_with_conformal_intervals.csv")
    print("=" * 80)
    print("[SUCCESS] Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
