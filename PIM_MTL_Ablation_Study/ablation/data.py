"""
Dataset Loading, Splitting, and Preprocessing for PIM-MTL Ablation Studies.

Guarantees:
- Fixed 70:30 train/test split with deterministic seed 42.
- Zero data leakage: all encoders and scalers are fitted strictly on training data.
- Preservation of missing-target masks.
- Clean feature ablation (dropping NDT features prior to preprocessing).
- Scientifically valid imputation strategies (fitted strictly on training data,
  while test labels remain strictly un-imputed ground truth).
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer

from ablation.config import DataConfig

CATEGORICAL_COLS = ["Type_of_cement", "Brand", "Additives", "Type_of_aggregates"]
FULL_NUMERICAL_COLS = ["Design_Fc_Mpa", "Curing_age_days", "Er_ohm-cm", "UPV_m/s"]
NO_NDT_NUMERICAL_COLS = ["Design_Fc_Mpa", "Curing_age_days"]
TARGET_COLS = ["Cs_Mpa", "Ts_Mpa", "Fs_Mpa"]


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load and clean dataset from CSV."""
    if not os.path.exists(data_path):
        # Fallback to capstone directory structure if relative path not found
        candidate_paths = [
            data_path,
            os.path.join(os.path.dirname(__file__), "..", data_path),
            r"C:\Users\jeetu\projects\Capstone\PIM_MTL_Ablation_Study\data\Data.csv",
            r"C:\Users\jeetu\projects\Capstone\3rdYearsem5\Data.csv",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                data_path = p
                break

    df = pd.read_csv(data_path)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def prepare_data(
    config: DataConfig,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, Any],
]:
    """
    Split, encode, scale, and handle missing targets according to DataConfig.

    Returns:
        (X_train, y_train_scaled, mask_train, y_train_raw),
        (X_test, y_test_scaled, mask_test, y_test_raw),
        preprocessor_dict
    """
    df = load_dataset(config.data_path)

    # 1. Fixed Train/Test Split (70:30 with seed 42)
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.split_seed,
    )
    test_indices = test_df.index.to_numpy()

    # Determine numerical features based on feature ablation config
    num_cols = FULL_NUMERICAL_COLS if config.include_ndt else NO_NDT_NUMERICAL_COLS

    # 2. Categorical One-Hot Encoding (Fit train only)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_train = ohe.fit_transform(train_df[CATEGORICAL_COLS])
    cat_test = ohe.transform(test_df[CATEGORICAL_COLS])

    # 3. Numerical Feature Scaling with RobustScaler (Fit train only)
    scaler_x = RobustScaler()
    num_train = scaler_x.fit_transform(train_df[num_cols])
    num_test = scaler_x.transform(test_df[num_cols])

    X_train = np.hstack([num_train, cat_train]).astype(np.float32)
    X_test = np.hstack([num_test, cat_test]).astype(np.float32)

    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = num_cols + cat_feature_names

    # 4. Target Extraction and Mask Creation
    y_train_raw = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
    y_test_raw = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

    mask_train = (~np.isnan(y_train_raw)).astype(np.float32)
    mask_test = (~np.isnan(y_test_raw)).astype(np.float32)

    # 5. Fit Target Scalers on Genuine Observed Training Samples
    target_scalers: Dict[str, RobustScaler] = {}
    target_centers = np.zeros(3, dtype=np.float32)
    target_scales = np.ones(3, dtype=np.float32)

    for i, col in enumerate(TARGET_COLS):
        sc = RobustScaler()
        obs_train = y_train_raw[mask_train[:, i] == 1.0, i].reshape(-1, 1)
        sc.fit(obs_train)
        target_scalers[col] = sc
        target_centers[i] = sc.center_[0]
        target_scales[i] = sc.scale_[0]

    # 6. Apply Missing-Target Strategy
    y_train_proc = np.copy(y_train_raw)
    mask_train_proc = np.copy(mask_train)
    X_train_proc = np.copy(X_train)

    strategy = config.missing_strategy.lower().strip()

    if strategy == "masked":
        # Proposed PIM-MTL: Leave unobserved targets at 0.0 placeholder with mask=0.0
        pass

    elif strategy == "complete_case":
        # Listwise deletion: keep only training specimens with all 3 targets observed
        complete_mask = (mask_train.sum(axis=1) == 3.0)
        X_train_proc = X_train_proc[complete_mask]
        y_train_proc = y_train_proc[complete_mask]
        mask_train_proc = mask_train_proc[complete_mask]

    elif strategy == "mean_imputation":
        # Compute mean strictly from observed training values
        for i in range(3):
            obs_vals = y_train_raw[mask_train[:, i] == 1.0, i]
            mean_val = float(np.mean(obs_vals))
            unobs_idx = (mask_train[:, i] == 0.0)
            y_train_proc[unobs_idx, i] = mean_val
        # Since missing values are imputed, mask is set to 1.0 so empirical MSE trains on all imputed values
        mask_train_proc = np.ones_like(y_train_proc, dtype=np.float32)

    elif strategy == "knn_imputation":
        # Impute missing training targets using KNN (k=5) based on X and observed y
        imputer_input = np.hstack([X_train, y_train_raw])
        imputer = KNNImputer(n_neighbors=config.knn_neighbors)
        imputed_matrix = imputer.fit_transform(imputer_input)
        y_train_proc = imputed_matrix[:, -3:].astype(np.float32)
        mask_train_proc = np.ones_like(y_train_proc, dtype=np.float32)

    else:
        raise ValueError(
            f"Unsupported missing_strategy: '{config.missing_strategy}'. "
            f"Choose from ['masked', 'complete_case', 'mean_imputation', 'knn_imputation']."
        )

    # 7. Scale Target Values to Standardized Space
    y_train_scaled = np.zeros_like(y_train_proc, dtype=np.float32)
    y_test_scaled = np.zeros_like(y_test_raw, dtype=np.float32)

    for i, col in enumerate(TARGET_COLS):
        sc = target_scalers[col]
        # For training: scale values that have non-zero mask (or all if imputed)
        active_train = (mask_train_proc[:, i] == 1.0)
        if active_train.any():
            y_train_scaled[active_train, i] = sc.transform(
                y_train_proc[active_train, i].reshape(-1, 1)
            ).ravel()

        # For test: scale observed values only; unobserved stay 0.0
        obs_test = (mask_test[:, i] == 1.0)
        if obs_test.any():
            y_test_scaled[obs_test, i] = sc.transform(
                y_test_raw[obs_test, i].reshape(-1, 1)
            ).ravel()

    preprocessor = {
        "ohe": ohe,
        "scaler_x": scaler_x,
        "target_scalers": target_scalers,
        "target_centers": target_centers,
        "target_scales": target_scales,
        "feature_names": all_feature_names,
        "input_dim": X_train_proc.shape[1],
        "n_train": len(X_train_proc),
        "n_test": len(X_test),
        "test_indices": test_indices,
    }

    return (
        (X_train_proc, y_train_scaled, mask_train_proc, y_train_proc),
        (X_test, y_test_scaled, mask_test, y_test_raw),
        preprocessor,
    )

