"""
Main Model Comparison and ACI Constraint Auditing Orchestrator.
Benchmarks all 5 baseline models from 4thsem against PIM-MTL from 3rdYearsem5
on a standardized 30% held-out test split (1,326 specimens).

Evaluates:
- Statistical accuracy: MSE, RMSE, R2, Adjusted R2, MAE for Cs, Ts, Fs
- Physical compliance: ACI 318 tensile/flexural envelopes, impossible Ts >= Cs, hierarchy Ts >= Fs
"""
import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Prevent Windows console encoding issues
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    ROOT_DIR,
    DATA_PATH,
    SAVED_MODELS_4THSEM,
    PIM_MTL_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURES,
    TARGET_VARIABLES,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_LABELS,
    ACI_BOUNDS,
)
from evaluate_constraints import compute_accuracy_metrics, audit_physical_constraints

# Add project directories to sys.path
sys.path.insert(0, str(PIM_MTL_DIR))
sys.path.insert(0, str(ROOT_DIR / "4thsem"))
from utils.models import PIM_MTL_Net
from utils.losses import PIM_MTL_Loss


def load_dataset():
    """Load and sanitize raw concrete dataset."""
    df = pd.read_csv(DATA_PATH)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def prepare_tensors(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encodes and scales inputs and targets for PIM-MTL networks."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_tr = ohe.fit_transform(train_df[CATEGORICAL_FEATURES])
    cat_te = ohe.transform(test_df[CATEGORICAL_FEATURES])

    scaler_x = RobustScaler()
    num_tr = scaler_x.fit_transform(train_df[NUMERICAL_FEATURES])
    num_te = scaler_x.transform(test_df[NUMERICAL_FEATURES])

    X_train = np.hstack([num_tr, cat_tr]).astype(np.float32)
    X_test = np.hstack([num_te, cat_te]).astype(np.float32)

    target_cols = [TARGET_VARIABLES["Cs"], TARGET_VARIABLES["Ts"], TARGET_VARIABLES["Fs"]]
    y_train_raw = train_df[target_cols].to_numpy(dtype=np.float32)
    mask_train = (~np.isnan(y_train_raw)).astype(np.float32)

    target_centers = np.zeros(3, dtype=np.float32)
    target_scales = np.ones(3, dtype=np.float32)
    y_train_scaled = np.zeros_like(y_train_raw)

    for i in range(3):
        sc = RobustScaler()
        obs = y_train_raw[mask_train[:, i] == 1.0, i].reshape(-1, 1)
        sc.fit(obs)
        target_centers[i] = sc.center_[0]
        target_scales[i] = sc.scale_[0]
        obs_idx = mask_train[:, i] == 1.0
        y_train_scaled[obs_idx, i] = sc.transform(y_train_raw[obs_idx, i].reshape(-1, 1)).ravel()

    return (
        X_train, y_train_scaled, mask_train,
        X_test, target_scales, target_centers,
    )


def train_standard_pim_mtl(X_train, y_train_scaled, mask_train, X_test, target_scales, target_centers, epochs: int = 60):
    """Train standard PIM-MTL model."""
    print("  -> Training PIM-MTL on 70% split (seed=42, 60 epochs)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model = PIM_MTL_Net(input_dim=X_train.shape[1]).to(device)
    criterion = PIM_MTL_Loss(
        lambda_phys=0.5,
        target_centers=target_centers,
        target_scales=target_scales,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        for bx, by, bm in loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss, _, _ = criterion(preds, by, bm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_preds_scaled = model(test_tensor).cpu().numpy()

    test_preds_raw = test_preds_scaled * target_scales + target_centers
    return {
        "model": model,
        "Cs": test_preds_raw[:, 0],
        "Ts": test_preds_raw[:, 1],
        "Fs": test_preds_raw[:, 2],
    }




def load_baseline_pipeline(target_key: str, model_type: str):
    """Load pre-trained and saved pipeline from 4thsem."""
    pkl_file = SAVED_MODELS_4THSEM / target_key / f"{model_type}_pipeline.pkl"
    if not pkl_file.exists():
        raise FileNotFoundError(f"Missing pipeline: {pkl_file}")
    with open(pkl_file, "rb") as f:
        return pickle.load(f)


def run_benchmark():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("      CONCRETE 3DCP: MODEL BENCHMARK & ACI 318 CONSTRAINT AUDITING SUITE")
    print("      Comparing 4thsem Baselines vs PIM-MTL (3rdYearsem5)")
    print("=" * 85)

    df = load_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"\n[Data Ingestion] Total records: {len(df)} | Train: {len(train_df)} (70%) | Test: {len(test_df)} (30%)")

    y_test_Cs = test_df[TARGET_VARIABLES["Cs"]].to_numpy(dtype=np.float64)
    y_test_Ts = test_df[TARGET_VARIABLES["Ts"]].to_numpy(dtype=np.float64)
    y_test_Fs = test_df[TARGET_VARIABLES["Fs"]].to_numpy(dtype=np.float64)

    # 1. Baseline Predictions
    model_predictions = {}
    baseline_types = ["ridge", "naive_bayes", "xgboost", "lightgbm", "ann"]
    X_test_raw = test_df[ALL_FEATURES]

    for m_type in baseline_types:
        model_name = MODEL_LABELS[m_type]
        print(f"\n[*] Generating predictions for baseline: {model_name}...")
        pipe_cs = load_baseline_pipeline("Cs", m_type)
        pipe_ts = load_baseline_pipeline("Ts", m_type)
        pipe_fs = load_baseline_pipeline("Fs", m_type)

        pred_cs = np.asarray(pipe_cs.predict(X_test_raw), dtype=np.float64)
        pred_ts = np.asarray(pipe_ts.predict(X_test_raw), dtype=np.float64)
        pred_fs = np.asarray(pipe_fs.predict(X_test_raw), dtype=np.float64)

        model_predictions[model_name] = {
            "key": m_type,
            "Cs": pred_cs,
            "Ts": pred_ts,
            "Fs": pred_fs,
        }

    # 2. PIM-MTL
    print("\n[*] Processing Physics-Informed Multi-Task Learning (PIM-MTL)...")
    X_train, y_train_scaled, mask_train, X_test, target_scales, target_centers = prepare_tensors(train_df, test_df)
    pim_std = train_standard_pim_mtl(X_train, y_train_scaled, mask_train, X_test, target_scales, target_centers)
    model_predictions[MODEL_LABELS["pim_mtl"]] = {
        "key": "pim_mtl",
        "Cs": pim_std["Cs"],
        "Ts": pim_std["Ts"],
        "Fs": pim_std["Fs"],
    }

    # 4. Accuracy Metrics
    accuracy_records = []
    print("\n" + "=" * 85)
    print("                    EMPIRICAL ACCURACY METRICS (30% TEST SET)")
    print("=" * 85)

    for model_name, preds in model_predictions.items():
        m_cs = compute_accuracy_metrics(y_test_Cs, preds["Cs"])
        m_ts = compute_accuracy_metrics(y_test_Ts, preds["Ts"])
        m_fs = compute_accuracy_metrics(y_test_Fs, preds["Fs"])

        for target_key, metric_dict in [("Cs", m_cs), ("Ts", m_ts), ("Fs", m_fs)]:
            accuracy_records.append({
                "Model": model_name,
                "Model_Key": preds["key"],
                "Target": target_key,
                "Observed_N": metric_dict["N"],
                "R2": metric_dict["R2"],
                "Adj_R2": metric_dict["Adj_R2"],
                "RMSE": metric_dict["RMSE"],
                "MAE": metric_dict["MAE"],
                "MSE": metric_dict["MSE"],
            })

    accuracy_df = pd.DataFrame(accuracy_records)
    print(accuracy_df[["Target", "Model", "R2", "Adj_R2", "RMSE", "MAE", "MSE"]].to_string(index=False))

    # 5. Physical & ACI 318 Constraint Auditing
    constraint_records = []
    print("\n" + "=" * 85)
    print("               PHYSICAL & ACI 318 CONSTRAINT COMPLIANCE AUDIT")
    print("=" * 85)

    for model_name, preds in model_predictions.items():
        audit = audit_physical_constraints(preds["Cs"], preds["Ts"], preds["Fs"])
        constraint_records.append({
            "Model": model_name,
            "Model_Key": preds["key"],
            **audit,
        })

    constraint_df = pd.DataFrame(constraint_records)
    audit_display_cols = [
        "Model",
        "Impossible_Rate_Pct",
        "Hierarchy_Viol_Rate_Pct",
        "ACI_Rt_Viol_Rate_Pct",
        "ACI_Rf_Viol_Rate_Pct",
        "Overall_Compliance_Rate_Pct",
    ]
    print(constraint_df[audit_display_cols].to_string(index=False))

    # 6. Combined Comprehensive Summary Table
    summary_rows = []
    for model_name in model_predictions.keys():
        sub_acc = accuracy_df[accuracy_df["Model"] == model_name].set_index("Target")
        sub_con = constraint_df[constraint_df["Model"] == model_name].iloc[0]

        summary_rows.append({
            "Model": model_name,
            "Cs_R2": sub_acc.loc["Cs", "R2"],
            "Cs_RMSE": sub_acc.loc["Cs", "RMSE"],
            "Ts_R2": sub_acc.loc["Ts", "R2"],
            "Ts_RMSE": sub_acc.loc["Ts", "RMSE"],
            "Fs_R2": sub_acc.loc["Fs", "R2"],
            "Fs_RMSE": sub_acc.loc["Fs", "RMSE"],
            "Impossible_Viol_Pct": sub_con["Impossible_Rate_Pct"],
            "Hierarchy_Viol_Pct": sub_con["Hierarchy_Viol_Rate_Pct"],
            "ACI_Rt_Viol_Pct": sub_con["ACI_Rt_Viol_Rate_Pct"],
            "ACI_Rf_Viol_Pct": sub_con["ACI_Rf_Viol_Rate_Pct"],
            "Overall_Compliance_Pct": sub_con["Overall_Compliance_Rate_Pct"],
        })
    summary_df = pd.DataFrame(summary_rows)

    # 7. Save full test predictions
    test_export_df = test_df[ALL_FEATURES].copy()
    test_export_df["True_Cs_Mpa"] = y_test_Cs
    test_export_df["True_Ts_Mpa"] = y_test_Ts
    test_export_df["True_Fs_Mpa"] = y_test_Fs

    for model_name, preds in model_predictions.items():
        k = preds["key"]
        test_export_df[f"Pred_Cs_{k}"] = preds["Cs"]
        test_export_df[f"Pred_Ts_{k}"] = preds["Ts"]
        test_export_df[f"Pred_Fs_{k}"] = preds["Fs"]
        test_export_df[f"Ratio_Rt_{k}"] = preds["Ts"] / np.maximum(preds["Cs"], 1e-4)
        test_export_df[f"Ratio_Rf_{k}"] = preds["Fs"] / np.maximum(preds["Cs"], 1e-4)

    # Export CSVs & Markdown Report
    accuracy_csv = RESULTS_DIR / "accuracy_metrics.csv"
    constraint_csv = RESULTS_DIR / "constraint_violations.csv"
    summary_csv = RESULTS_DIR / "comprehensive_summary.csv"
    predictions_csv = RESULTS_DIR / "full_test_predictions.csv"
    report_md = RESULTS_DIR / "comparison_report.md"

    accuracy_df.to_csv(accuracy_csv, index=False)
    constraint_df.to_csv(constraint_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    test_export_df.to_csv(predictions_csv, index=False)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Model Comparison & Physical Constraint Audit Report\n\n")
        f.write(f"**Test Set Size:** {len(test_df)} specimens (30% split, random_state=42)\n\n")
        f.write("## 1. Overall Performance and Physical Compliance Summary\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        f.write("## 2. Detailed Empirical Accuracy Metrics\n\n")
        f.write(accuracy_df[["Target", "Model", "R2", "Adj_R2", "RMSE", "MAE", "MSE"]].to_markdown(index=False) + "\n\n")
        f.write("## 3. Physical & ACI 318-19 Constraint Compliance\n\n")
        f.write(constraint_df[audit_display_cols].to_markdown(index=False) + "\n\n")

    print(f"\n[Artifacts Generated Successfully]:")
    print(f"  - {accuracy_csv}")
    print(f"  - {constraint_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {predictions_csv}")
    print(f"  - {report_md}")

    # 8. Trigger Plotting Suite
    print("\n[*] Generating Publication-Quality Figures...")
    from plot_comparisons import generate_all_plots
    generate_all_plots(accuracy_df, constraint_df, summary_df, test_export_df)

    print(f"\nExecution finished in {time.time() - start_time:.2f} seconds.")
    return summary_df


if __name__ == "__main__":
    run_benchmark()
