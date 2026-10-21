"""
Main Entry Point for PIM-MTL Training and Inference.

Provides a simplified, minimal workflow to train the Physics-Informed Masked
Multi-Task Neural Network on ConcreteXAI and save the trained model to models/.
"""

import argparse
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils.models import PIM_MTL_Net
from utils.losses import PIM_MTL_Loss

# Core schema constants
CATEGORICAL_COLS = ["Type_of_cement", "Brand", "Additives", "Type_of_aggregates"]
NUMERICAL_COLS = ["Design_Fc_Mpa", "Curing_age_days", "Er_ohm-cm", "UPV_m/s"]
TARGET_COLS = ["Cs_Mpa", "Ts_Mpa", "Fs_Mpa"]
SEED = 42


def set_seed(seed: int = SEED):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_preprocess(data_path: str):
    """Load dataset, encode categorical variables, and scale features and targets."""
    df = pd.read_csv(data_path)

    # Clean whitespace in strings
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # Categorical One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_train = ohe.fit_transform(train_df[CATEGORICAL_COLS])
    cat_test = ohe.transform(test_df[CATEGORICAL_COLS])

    # Numerical Feature Scaling
    scaler_x = RobustScaler()
    num_train = scaler_x.fit_transform(train_df[NUMERICAL_COLS])
    num_test = scaler_x.transform(test_df[NUMERICAL_COLS])

    X_train = np.hstack([num_train, cat_train]).astype(np.float32)
    X_test = np.hstack([num_test, cat_test]).astype(np.float32)

    # Target Extraction and Scaling
    y_train_raw = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
    y_test_raw = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

    mask_train = (~np.isnan(y_train_raw)).astype(np.float32)
    mask_test = (~np.isnan(y_test_raw)).astype(np.float32)

    target_scalers = {}
    target_centers = np.zeros(3, dtype=np.float32)
    target_scales = np.ones(3, dtype=np.float32)
    y_train_scaled = np.zeros_like(y_train_raw)
    y_test_scaled = np.zeros_like(y_test_raw)

    for i, col in enumerate(TARGET_COLS):
        sc = RobustScaler()
        obs_train = y_train_raw[mask_train[:, i] == 1.0, i].reshape(-1, 1)
        sc.fit(obs_train)
        target_scalers[col] = sc
        target_centers[i] = sc.center_[0]
        target_scales[i] = sc.scale_[0]

        # Scale observed values; unobserved stay 0.0
        obs_idx_tr = mask_train[:, i] == 1.0
        y_train_scaled[obs_idx_tr, i] = sc.transform(y_train_raw[obs_idx_tr, i].reshape(-1, 1)).ravel()

        obs_idx_te = mask_test[:, i] == 1.0
        y_test_scaled[obs_idx_te, i] = sc.transform(y_test_raw[obs_idx_te, i].reshape(-1, 1)).ravel()

    preprocessor = {
        "ohe": ohe,
        "scaler_x": scaler_x,
        "target_scalers": target_scalers,
        "target_centers": target_centers,
        "target_scales": target_scales,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    return (
        (X_train, y_train_scaled, mask_train, y_train_raw),
        (X_test, y_test_scaled, mask_test, y_test_raw),
        preprocessor,
    )


def train_model(X_train, y_train, mask_train, preprocessor, epochs=60, batch_size=64, lr=1e-3):
    """Train PIM_MTL_Net with custom physics-informed masked loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = PIM_MTL_Net(input_dim=input_dim).to(device)
    criterion = PIM_MTL_Loss(
        lambda_phys=0.5,
        target_centers=preprocessor["target_centers"],
        target_scales=preprocessor["target_scales"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting training on {device} (Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr})...")
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, total_data, total_phys = 0.0, 0.0, 0.0
        for bx, by, bm in loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)

            optimizer.zero_grad()
            preds = model(bx)
            loss, l_data, l_phys = criterion(preds, by, bm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(bx)
            total_data += l_data.item() * len(bx)
            total_phys += l_phys.item() * len(bx)

        scheduler.step()
        n_samples = len(X_train)
        avg_loss = total_loss / n_samples
        avg_data = total_data / n_samples
        avg_phys = total_phys / n_samples
        history.append((epoch, avg_loss, avg_data, avg_phys))

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | Total Loss: {avg_loss:.4f} | "
                f"Data MSE: {avg_data:.4f} | Phys Penalty: {avg_phys:.6f}"
            )

    return model, history


def evaluate_model(model, X_test, y_test_raw, mask_test, preprocessor, results_path=None):
    """Evaluate model predictions on unscaled original test targets and save report."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    centers = preprocessor["target_centers"]
    scales = preprocessor["target_scales"]
    preds_raw = preds_scaled * scales + centers

    n_test = len(y_test_raw)
    metrics_summary = []

    print("\n" + "=" * 70)
    print("           PIM-MTL TEST SET QUANTITATIVE EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Target':<10} | {'Observed N':<10} | {'Coverage':<10} | {'R2 Score':<10} | {'RMSE (MPa)':<12} | {'MAE (MPa)':<10}")
    print("-" * 70)

    for i, col in enumerate(TARGET_COLS):
        valid = mask_test[:, i] == 1.0
        n_obs = int(valid.sum())
        cov_pct = (n_obs / n_test) * 100.0
        y_true = y_test_raw[valid, i]
        y_pred = preds_raw[valid, i]

        r2 = r2_score(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        metrics_summary.append({
            "target": col,
            "observed_n": n_obs,
            "coverage_pct": cov_pct,
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        })

        print(f"{col:<10} | {n_obs:<10} | {cov_pct:>8.1f}% | {r2:>9.4f}  | {rmse:>10.4f}   | {mae:>8.4f}")

    print("-" * 70)

    # Physical Violation Rate (PVR) Evaluation
    cs_pred = np.maximum(preds_raw[:, 0], 1.0)
    ts_pred = preds_raw[:, 1]
    fs_pred = preds_raw[:, 2]
    rt = ts_pred / cs_pred
    rf = fs_pred / cs_pred

    pvr_ts = float(np.mean((rt < 0.06) | (rt > 0.18)) * 100.0)
    pvr_fs = float(np.mean((rf < 0.08) | (rf > 0.25)) * 100.0)
    pvr_impossible = float(np.mean(ts_pred >= cs_pred) * 100.0)

    print("\n--- Physical Feasibility & Constraint Audit ---")
    print(f"1. Tensile Ratio Compliance [0.06 <= Ts/Cs <= 0.18]   : Violation Rate = {pvr_ts:.2f}% (Compliance = {100-pvr_ts:.2f}%)")
    print(f"2. Flexural Ratio Compliance [0.08 <= Fs/Cs <= 0.25]  : Violation Rate = {pvr_fs:.2f}% (Compliance = {100-pvr_fs:.2f}%)")
    print(f"3. Physically Impossible Preds (Ts >= Cs)             : Violation Rate = {pvr_impossible:.2f}% (Compliance = {100-pvr_impossible:.2f}%)")
    print("=" * 70)

    if results_path:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("      PIM-MTL CONCRETE STRENGTH PREDICTION PIPELINE RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Test Specimens: {n_test}\n")
            f.write(f"Compute Device      : {device}\n\n")
            f.write("--- Quantitative Accuracy Metrics ---\n")
            f.write(f"{'Target':<10} | {'Observed N':<10} | {'Coverage':<10} | {'R2 Score':<10} | {'RMSE (MPa)':<12} | {'MAE (MPa)':<10}\n")
            f.write("-" * 70 + "\n")
            for m in metrics_summary:
                f.write(f"{m['target']:<10} | {m['observed_n']:<10} | {m['coverage_pct']:>8.1f}% | {m['r2']:>9.4f}  | {m['rmse']:>10.4f}   | {m['mae']:>8.4f}\n")
            f.write("-" * 70 + "\n\n")
            f.write("--- Physical Feasibility Audit ---\n")
            f.write(f"Tensile Ratio Violations (Rt outside [0.06, 0.18])  : {pvr_ts:.2f}%\n")
            f.write(f"Flexural Ratio Violations (Rf outside [0.08, 0.25]) : {pvr_fs:.2f}%\n")
            f.write(f"Physically Impossible Predictions (Ts >= Cs)        : {pvr_impossible:.2f}%\n\n")
            f.write("=" * 70 + "\n")
        print(f"\n[INFO] Comprehensive pipeline results successfully saved to: {results_path}")

    return metrics_summary, {"pvr_ts": pvr_ts, "pvr_fs": pvr_fs, "pvr_impossible": pvr_impossible}


def main():
    parser = argparse.ArgumentParser(description="PIM-MTL Concrete Strength Model Runner")
    parser.add_argument("--data", default="Data.csv", help="Path to Data.csv")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", default="models/pim_mtl_model.pt", help="Path to save trained model")
    parser.add_argument("--results-file", default="results.txt", help="Path to save results text report")
    args = parser.parse_args()

    set_seed(SEED)

    train_data, test_data, preprocessor = load_and_preprocess(args.data)
    X_train, y_train, mask_train, _ = train_data
    X_test, _, mask_test, y_test_raw = test_data

    model, history = train_model(
        X_train, y_train, mask_train, preprocessor, epochs=args.epochs, batch_size=args.batch_size
    )
    evaluate_model(model, X_test, y_test_raw, mask_test, preprocessor, results_path=args.results_file)

    # Save model and preprocessor artifacts
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model.save(args.output)
    preproc_path = os.path.join(os.path.dirname(args.output), "preprocessor.joblib")
    joblib.dump(preprocessor, preproc_path)
    print(f"Trained model saved to {args.output}")
    print(f"Preprocessor saved to {preproc_path}")


if __name__ == "__main__":
    main()
