"""
Inference Module for Mechanical Strength Prediction.
Loads serialized best pipelines for EACH model type (or best overall)
and generates predictions on new concrete mix data from CSV/Excel files or DataFrames.
"""
import argparse
import sys
from pathlib import Path
from typing import Union, Optional, List
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import SAVED_MODELS_DIR, ALL_FEATURES, CATEGORICAL_FEATURES
from test_models import load_model, AVAILABLE_MODEL_TYPES


class ConcreteStrengthPredictor:
    """
    Unified predictor that loads models for Cs, Ts, and Fs
    with support for selecting specific model architectures:
    'best' (default), 'xgboost', 'lightgbm', 'ann', 'ridge', or 'naive_bayes'.
    """
    def __init__(
        self,
        model_type: str = "best",
        models_dir: Union[str, Path] = SAVED_MODELS_DIR,
    ):
        self.model_type = model_type.lower()
        self.models_dir = Path(models_dir)
        self.models = {}
        self._load_models()

    def _load_models(self):
        targets = ["Cs", "Ts", "Fs"]
        for target in targets:
            try:
                self.models[target] = load_model(target, self.model_type, self.models_dir)
            except Exception as e:
                # Fallback to root or best if specific type not found
                try:
                    self.models[target] = load_model(target, "best", self.models_dir)
                except Exception:
                    pass

        if not self.models:
            raise FileNotFoundError(
                f"No saved models found in {self.models_dir}. "
                "Please run `train.py` first to train and serialize models."
            )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for Cs, Ts, and Fs on input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing input features.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with predicted columns:
            ['Predicted_Cs_Mpa', 'Predicted_Ts_Mpa', 'Predicted_Fs_Mpa']
        """
        # Validate required features
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required feature columns: {missing}")

        data = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()

        X = data[ALL_FEATURES]
        output_df = data.copy()

        suffix = f"_{self.model_type}" if self.model_type != "best" else ""
        if "Cs" in self.models:
            output_df[f"Predicted_Cs_Mpa{suffix}"] = self.models["Cs"].predict(X)
        if "Ts" in self.models:
            output_df[f"Predicted_Ts_Mpa{suffix}"] = self.models["Ts"].predict(X)
        if "Fs" in self.models:
            output_df[f"Predicted_Fs_Mpa{suffix}"] = self.models["Fs"].predict(X)

        return output_df

    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load CSV/Excel file, generate predictions, and optionally save results."""
        input_p = Path(input_path)
        if input_p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_p)
        else:
            df = pd.read_csv(input_p)

        predictions_df = self.predict(df)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() in [".xlsx", ".xls"]:
                predictions_df.to_excel(out_p, index=False)
            else:
                predictions_df.to_csv(out_p, index=False)
            print(f"Predictions saved successfully to: {out_p}")

        return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Predict Concrete Mechanical Strength (Cs, Ts, Fs)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input CSV or Excel file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save predictions")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="best",
        choices=AVAILABLE_MODEL_TYPES,
        help="Model architecture to use ('best', 'xgboost', 'lightgbm', 'ann', 'ridge', 'naive_bayes')",
    )
    args = parser.parse_args()

    predictor = ConcreteStrengthPredictor(model_type=args.model)

    if args.input:
        preds = predictor.predict_file(args.input, args.output)
        print(f"\nSample Predictions (using {args.model.upper()}):")
        target_cols = [c for c in preds.columns if c.startswith("Predicted_")]
        print(preds[ALL_FEATURES + target_cols].head())
    else:
        # Run a demonstration on sample rows from Data.csv
        from dataset import load_raw_data
        sample_df = load_raw_data().head(5)
        preds = predictor.predict(sample_df)
        print(f"\nDemonstration Prediction using {args.model.upper()} on first 5 rows of Data.csv:")
        pred_cols = [c for c in preds.columns if c.startswith("Predicted_")]
        print(preds[["Design_Fc_Mpa", "Curing_age_days"] + pred_cols])


if __name__ == "__main__":
    main()
