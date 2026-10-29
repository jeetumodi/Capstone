"""
Main Training, Evaluation, and Model Serialization Script.
Trains and hyperparameter-tunes all 5 models across all 3 targets:
1. Compressive Strength (Cs)
2. Tensile Strength (Ts)
3. Flexural Strength (Fs)

Model Types per target:
- Ridge Regression
- Custom Naive Bayes Regressor (20 quantile bins)
- XGBoost Regressor
- LightGBM Regressor
- PyTorch Artificial Neural Network (ANN)

Serialization Strategy:
Saves each of the 5 model pipelines per target in `saved_models/{target}/`
so that EVERY model type can be loaded and tested on independently:
- `ridge_pipeline.pkl`
- `naive_bayes_pipeline.pkl`
- `xgboost_pipeline.pkl`
- `lightgbm_pipeline.pkl`
- `ann_pipeline.pkl` (+ `ann_weights.pt`)
- `best_overall_model.pkl`
- `metrics_summary.json`
Also saves the held-out 30% test sets in `saved_models/test_sets/` for instant re-testing.
"""
import sys
import json
import pickle
import time
from pathlib import Path
import pandas as pd
import torch

# Prevent Unicode encoding issues on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    SAVED_MODELS_DIR,
    TARGET_VARIABLES,
    TARGET_DESCRIPTIONS,
    ALL_FEATURES,
)
from dataset import load_raw_data, prepare_data_for_target
from trainer import (
    train_ridge,
    train_naive_bayes,
    train_xgboost,
    train_lightgbm,
    train_ann,
)
from evaluate import evaluate_model, format_results_table


MODEL_FILE_KEYS = {
    "Ridge Regression": "ridge",
    "Naive Bayes": "naive_bayes",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "ANN (PyTorch)": "ann",
}


def run_pipeline():
    start_total_time = time.time()
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    test_sets_dir = SAVED_MODELS_DIR / "test_sets"
    test_sets_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  INTELLIGENT 3DCP: MECHANICAL STRENGTH PREDICTION PIPELINE")
    print("  Models: Ridge | Naive Bayes | XGBoost | LightGBM | PyTorch ANN")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  PyTorch Compute Device: {device_name}")
    print("=" * 80)

    # 1. Load Data
    raw_df = load_raw_data()
    print(f"\n[Data Loaded] Total records in Data.csv: {len(raw_df)}")

    all_test_metrics = []
    overall_best_models = {}

    # 2. Iterate through each strength target
    for target_key, target_col in TARGET_VARIABLES.items():
        target_name = TARGET_DESCRIPTIONS[target_key]
        print("\n" + "#" * 80)
        print(f"  PROCESSING TARGET: {target_name} ({target_col})")
        print("#" * 80)

        # Prepare 70/30 train/test split
        X_train, X_test, y_train, y_test = prepare_data_for_target(raw_df, target_col)
        n_features = X_train.shape[1]
        print(f"  Training samples (70%): {len(X_train)} | Test samples (30%): {len(X_test)}")

        # Save the held-out test split for convenient future loading and testing
        test_df = X_test.copy()
        test_df[target_col] = y_test
        test_df.to_csv(test_sets_dir / f"test_{target_key}.csv", index=False)

        target_trained_models = {}

        # 1. Ridge Regression
        t0 = time.time()
        ridge_pipe, ridge_params, ridge_cv = train_ridge(X_train, y_train)
        ridge_metrics = evaluate_model(ridge_pipe, X_test, y_test, n_features=n_features)
        target_trained_models["Ridge Regression"] = {
            "pipeline": ridge_pipe,
            "params": ridge_params,
            "cv_score": ridge_cv,
            "metrics": ridge_metrics,
            "fit_time": time.time() - t0,
        }

        # 2. Naive Bayes Regressor
        t0 = time.time()
        nb_pipe, nb_params, nb_cv = train_naive_bayes(X_train, y_train)
        nb_metrics = evaluate_model(nb_pipe, X_test, y_test, n_features=n_features)
        target_trained_models["Naive Bayes"] = {
            "pipeline": nb_pipe,
            "params": nb_params,
            "cv_score": nb_cv,
            "metrics": nb_metrics,
            "fit_time": time.time() - t0,
        }

        # 3. XGBoost Regressor
        t0 = time.time()
        xgb_pipe, xgb_params, xgb_cv = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_pipe, X_test, y_test, n_features=n_features)
        target_trained_models["XGBoost"] = {
            "pipeline": xgb_pipe,
            "params": xgb_params,
            "cv_score": xgb_cv,
            "metrics": xgb_metrics,
            "fit_time": time.time() - t0,
        }

        # 4. LightGBM Regressor
        t0 = time.time()
        lgb_pipe, lgb_params, lgb_cv = train_lightgbm(X_train, y_train)
        lgb_metrics = evaluate_model(lgb_pipe, X_test, y_test, n_features=n_features)
        target_trained_models["LightGBM"] = {
            "pipeline": lgb_pipe,
            "params": lgb_params,
            "cv_score": lgb_cv,
            "metrics": lgb_metrics,
            "fit_time": time.time() - t0,
        }

        # 5. PyTorch ANN
        t0 = time.time()
        ann_pipe, ann_params, ann_cv = train_ann(X_train, y_train)
        ann_metrics = evaluate_model(ann_pipe, X_test, y_test, n_features=n_features)
        target_trained_models["ANN (PyTorch)"] = {
            "pipeline": ann_pipe,
            "params": ann_params,
            "cv_score": ann_cv,
            "metrics": ann_metrics,
            "fit_time": time.time() - t0,
        }

        # Tabulate test evaluation metrics for this target
        for model_name, info in target_trained_models.items():
            record = {
                "Target": target_key,
                "Target_Description": target_name,
                "Model": model_name,
                "CV_R2": info["cv_score"],
                "R2": info["metrics"]["R2"],
                "Adj_R2": info["metrics"]["Adj_R2"],
                "RMSE": info["metrics"]["RMSE"],
                "MAE": info["metrics"]["MAE"],
                "MSE": info["metrics"]["MSE"],
                "Fit_Time_Sec": round(info["fit_time"], 2),
            }
            all_test_metrics.append(record)

        # -------------------------------------------------------------
        # SERIALIZE BEST OF EACH MODEL TYPE PER TARGET
        # -------------------------------------------------------------
        target_dir = SAVED_MODELS_DIR / target_key
        target_dir.mkdir(parents=True, exist_ok=True)
        target_metadata = {
            "target_key": target_key,
            "target_column": target_col,
            "target_description": target_name,
            "models": {},
        }

        print(f"\n  [*] Saving individual best models for {target_name} to: {target_dir}")
        for model_name, info in target_trained_models.items():
            file_prefix = MODEL_FILE_KEYS[model_name]
            pipe_path = target_dir / f"{file_prefix}_pipeline.pkl"
            
            with open(pipe_path, "wb") as f:
                pickle.dump(info["pipeline"], f)
            print(f"      - Saved {model_name:<18} -> {pipe_path.name}")

            # If ANN, also save PyTorch checkpoint file
            if "ANN" in model_name:
                ann_reg = info["pipeline"].named_steps["regressor"]
                ann_pt_path = target_dir / "ann_weights.pt"
                ann_reg.save_checkpoint(str(ann_pt_path))
                print(f"        (Also saved PyTorch weights -> {ann_pt_path.name})")

            target_metadata["models"][file_prefix] = {
                "model_name": model_name,
                "pipeline_file": pipe_path.name,
                "best_parameters": {str(k): str(v) for k, v in info["params"].items()},
                "cv_score": info["cv_score"],
                "test_metrics": info["metrics"],
                "fit_time_seconds": round(info["fit_time"], 2),
            }

        # Determine and save overall best model for this target
        best_model_name = max(
            target_trained_models.keys(),
            key=lambda m: target_trained_models[m]["metrics"]["R2"],
        )
        best_info = target_trained_models[best_model_name]
        best_overall_path = target_dir / "best_overall_model.pkl"
        with open(best_overall_path, "wb") as f:
            pickle.dump(best_info["pipeline"], f)

        # Also save at root of saved_models for backwards compatibility
        root_best_path = SAVED_MODELS_DIR / f"best_pipeline_{target_key}.pkl"
        with open(root_best_path, "wb") as f:
            pickle.dump(best_info["pipeline"], f)

        target_metadata["best_overall_model"] = {
            "model_name": best_model_name,
            "pipeline_file": best_overall_path.name,
            "metrics": best_info["metrics"],
        }
        overall_best_models[target_key] = target_metadata["best_overall_model"]

        with open(target_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(target_metadata, f, indent=4)

        print(f"\n  [★ BEST OVERALL MODEL for {target_name}]: {best_model_name}")
        print(f"     Test R2: {best_info['metrics']['R2']:.4f} | Adj R2: {best_info['metrics']['Adj_R2']:.4f}")
        print(f"     RMSE: {best_info['metrics']['RMSE']:.4f} | MAE: {best_info['metrics']['MAE']:.4f} | MSE: {best_info['metrics']['MSE']:.4f}")

    # 3. Overall Summary Table
    results_df = format_results_table(all_test_metrics)
    summary_csv_path = SAVED_MODELS_DIR / "evaluation_summary.csv"
    results_df.to_csv(summary_csv_path, index=False)

    summary_md_path = SAVED_MODELS_DIR / "evaluation_summary.md"
    try:
        results_df.to_markdown(summary_md_path, index=False)
    except Exception:
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(results_df.to_string(index=False))

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 90)
    print("                    FINAL PERFORMANCE COMPARISON (30% TEST SET)")
    print("=" * 90)
    display_cols = ["Target", "Model", "R2", "Adj_R2", "RMSE", "MAE", "MSE", "Fit_Time_Sec"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 90)
    print(f"\nAll models saved successfully across all categories!")
    print(f"Total training & tuning time: {total_elapsed:.2f} seconds.")
    print(f"Saved models root directory: {SAVED_MODELS_DIR}")
    return results_df, overall_best_models


if __name__ == "__main__":
    run_pipeline()
