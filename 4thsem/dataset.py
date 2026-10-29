"""
Dataset loading and train/test splitting module for concrete strength parameters.
Performs clean 70/30 train/test splitting with fixed random_state=42.
Handles separate instances for Cs, Ts, and Fs without data leakage.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    DATA_PATH,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_raw_data(data_path=DATA_PATH) -> pd.DataFrame:
    """Load the concrete dataset and sanitize string columns."""
    df = pd.read_csv(data_path)
    
    # Strip any trailing/leading whitespace in categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df


def prepare_data_for_target(df: pd.DataFrame, target_col: str):
    """
    Extract features and target for a given strength variable,
    dropping rows where that specific target is missing, and
    splitting into 70% Train and 30% Test with random_state=42.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full loaded dataset.
    target_col : str
        Target column name (e.g. 'Cs_Mpa', 'Ts_Mpa', 'Fs_Mpa').
        
    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    # Filter valid instances for this target
    valid_df = df.dropna(subset=[target_col]).copy()
    
    X = valid_df[ALL_FEATURES]
    y = valid_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = load_raw_data()
    print(f"Total raw rows: {len(raw_df)}")
    for name, col in [("Compressive (Cs)", "Cs_Mpa"), ("Tensile (Ts)", "Ts_Mpa"), ("Flexural (Fs)", "Fs_Mpa")]:
        X_tr, X_te, y_tr, y_te = prepare_data_for_target(raw_df, col)
        print(f"[{name}] Valid samples: {len(X_tr) + len(X_te)} | Train: {len(X_tr)} (70%) | Test: {len(X_te)} (30%)")

