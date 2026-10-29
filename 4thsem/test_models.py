"""
Model Loading and Testing Module.
Allows loading and testing the best model for EACH model type across all targets:
- Targets: Cs, Ts, Fs
- Model Types:
  * 'ridge'
  * 'naive_bayes'
  * 'xgboost'
  * 'lightgbm'
  * 'ann'
  * 'best' (overall best model for the target)

Can evaluate on standard held-out test sets or custom user-provided datasets.
"""
import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    SAVED_MODELS_DIR,
    TARGET_VARIABLES,
    TARGET_DESCRIPTIONS,
    ALL_FEATURES,
)
from evaluate import calculate_metrics, format_results_table

AVAILABLE_MODEL_TYPES = [
    "ridge",
    "naive_bayes",
    "xgboost",
    "lightgbm",
    "ann",
    "best",
]

MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "naive_bayes": "Naive Bayes Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
    "ann": "PyTorch ANN Regressor",
    "best": "Best Overall Model",
}


def get_model_path(target: str, model_type: str, base_dir: Path = SAVED_MODELS_DIR) -> Path:
    """Resolve file path for a given target and model type."""
    target_clean = target.replace("_Mpa", "").capitalize()
    if target_clean not in ["Cs", "Ts", "Fs"]:
        # Match case-insensitively
        for t in ["Cs", "Ts", "Fs"]:
            if t.lower() == target.lower():
                target_clean = t
                break
        else:
            raise ValueError(f"Unknown target '{target}'. Expected one of ['Cs', 'Ts', 'Fs'].")

    model_type_clean = model_type.lower()
    if model_type_clean == "best":
        filename = "best_overall_model.pkl"
    else:
        if model_type_clean not in AVAILABLE_MODEL_TYPES:
            raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {AVAILABLE_MODEL_TYPES}.")
        filename = f"{model_type_clean}_pipeline.pkl"

    path = base_dir / target_clean / filename
    return path


def load_model(target: str, model_type: str = "best", base_dir: Path = SAVED_MODELS_DIR):
    """
    Load a trained and serialized end-to-end pipeline.

    Parameters
    ----------
    target : str
        Target property ('Cs', 'Ts', or 'Fs').
    model_type : str
        Model architecture ('ridge', 'naive_bayes', 'xgboost', 'lightgbm', 'ann', 'best').
    base_dir : Path
        Base directory containing saved_models.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline with preprocessing and fitted regressor.
    """
    model_path = get_model_path(target, model_type, base_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model pipeline not found at: {model_path}\n"
            "Please run `train.py` first to train and serialize the models."
        )

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def load_test_dataset(target: str, base_dir: Path = SAVED_MODELS_DIR) -> pd.DataFrame:
    """Load the held-out 30% test dataset for a given target."""
    target_clean = target.replace("_Mpa", "").capitalize()
    test_path = base_dir / "test_sets" / f"test_{target_clean}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")
    return pd.read_csv(test_path)


def test_single_model(
    target: str,
    model_type: str,
    test_df: Optional[pd.DataFrame] = None,
    base_dir: Path = SAVED_MODELS_DIR,
) -> Dict[str, Any]:
    """
    Evaluate a specific saved model type on a test dataset.
    """
    target_clean = target.replace("_Mpa", "").capitalize()
    target_col = TARGET_VARIABLES[target_clean]

    if test_df is None:
        test_df = load_test_dataset(target_clean, base_dir)

    pipeline = load_model(target_clean, model_type, base_dir)

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[target_col].values

    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, n_features=len(ALL_FEATURES))

    return {
        "Target": target_clean,
        "Target_Col": target_col,
        "Model_Type": model_type,
        "Model_Name": MODEL_DISPLAY_NAMES.get(model_type, model_type),
        "Test_Samples": len(test_df),
        **metrics,
    }


def test_all_saved_models(base_dir: Path = SAVED_MODELS_DIR) -> pd.DataFrame:
    """
    Evaluates every saved model type (ridge, naive_bayes, xgboost, lightgbm, ann)
    plus the best overall model across all three targets.
    """
    all_results = []
    print("=" * 85)
    print("      EVALUATING ALL SAVED MODEL TYPES ACROSS TARGETS (HELD-OUT 30% TEST SET)")
    print("=" * 85)

    for target_key in ["Cs", "Ts", "Fs"]:
        target_name = TARGET_DESCRIPTIONS[target_key]
        print(f"\n[*] Testing models for: {target_name}")
        test_df = load_test_dataset(target_key, base_dir)

        for m_type in ["ridge", "naive_bayes", "xgboost", "lightgbm", "ann", "best"]:
            try:
                res = test_single_model(target_key, m_type, test_df=test_df, base_dir=base_dir)
                all_results.append(res)
                print(
                    f"    {res['Model_Name']:<23} | R2: {res['R2']:.4f} | "
                    f"Adj_R2: {res['Adj_R2']:.4f} | RMSE: {res['RMSE']:.4f} MPa | "
                    f"MAE: {res['MAE']:.4f} MPa | MSE: {res['MSE']:.4f}"
                )
            except Exception as e:
                print(f"    {m_type:<23} | Failed to evaluate: {e}")

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 85)
    print("                              EVALUATION MATRIX")
    print("=" * 85)
    display_cols = ["Target", "Model_Name", "R2", "Adj_R2", "RMSE", "MAE", "MSE"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 85)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Test and Evaluate Saved Concrete Models")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default=None,
        choices=["Cs", "Ts", "Fs", "all"],
        help="Target strength property to evaluate",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=AVAILABLE_MODEL_TYPES + ["all"],
        help="Model type to test ('ridge', 'naive_bayes', 'xgboost', 'lightgbm', 'ann', 'best', 'all')",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Path to custom test CSV/Excel file (optional)",
    )
    args = parser.parse_args()

    custom_test = None
    if args.data:
        custom_p = Path(args.data)
        custom_test = pd.read_excel(custom_p) if custom_p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(custom_p)

    if args.target is None or args.target.lower() == "all":
        test_all_saved_models()
    else:
        target = args.target.capitalize()
        if args.model == "all":
            for m in ["ridge", "naive_bayes", "xgboost", "lightgbm", "ann", "best"]:
                res = test_single_model(target, m, test_df=custom_test)
                print(
                    f"[{target}] {res['Model_Name']:<23} | R2: {res['R2']:.4f} | "
                    f"Adj_R2: {res['Adj_R2']:.4f} | RMSE: {res['RMSE']:.4f} | "
                    f"MAE: {res['MAE']:.4f} | MSE: {res['MSE']:.4f}"
                )
        else:
            res = test_single_model(target, args.model, test_df=custom_test)
            print(f"\n--- Evaluation Results for {target} ({res['Model_Name']}) ---")
            for k in ["R2", "Adj_R2", "RMSE", "MAE", "MSE", "Test_Samples"]:
                print(f"  {k:<14}: {res[k]}")


if __name__ == "__main__":
    main()

