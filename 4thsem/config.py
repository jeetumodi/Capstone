"""
Configuration file for the 3DCP Concrete Mechanical Strength Prediction Pipeline.
Defines column mappings, hyperparameter grids, training parameters, and file paths.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data.csv"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# Target Variables (trained separately)
TARGET_VARIABLES = {
    "Cs": "Cs_Mpa",  # Compressive Strength (MPa)
    "Ts": "Ts_Mpa",  # Tensile Strength (MPa)
    "Fs": "Fs_Mpa",  # Flexural Strength (MPa)
}

TARGET_DESCRIPTIONS = {
    "Cs": "Compressive Strength (Cs_Mpa)",
    "Ts": "Tensile Strength (Ts_Mpa)",
    "Fs": "Flexural Strength (Fs_Mpa)",
}

# Feature Definitions
CATEGORICAL_FEATURES = [
    "Type_of_cement",
    "Brand",
    "Additives",
    "Type_of_aggregates",
]

NUMERICAL_FEATURES = [
    "Design_Fc_Mpa",
    "Curing_age_days",
    "Er_ohm-cm",
    "UPV_m/s",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Train/Test Split
TEST_SIZE = 0.30
TRAIN_SIZE = 0.70
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter Grids for 5-Fold Cross-Validation
# 1. Ridge Regression
RIDGE_PARAM_GRID = {
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

# 2. Naive Bayes Regressor
NAIVE_BAYES_N_BINS = 20

# 3. XGBoost Regressor
XGB_PARAM_GRID = {
    "regressor__max_depth": [4, 6],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
}
XGB_FIXED_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# 4. LightGBM Regressor
LGBM_PARAM_GRID = {
    "regressor__num_leaves": [31, 63],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
}
LGBM_FIXED_PARAMS = {
    "n_estimators": 300,
    "subsample_freq": 1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": 1,  # Keep single-thread inside worker to prevent lockups in CV
}

# 5. PyTorch ANN Regressor
ANN_GRID = [
    {"H": 128, "dropout_p": 0.2},
    {"H": 128, "dropout_p": 0.3},
    {"H": 256, "dropout_p": 0.2},
    {"H": 256, "dropout_p": 0.3},
]
ANN_TRAINING_CONFIG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "patience": 25,
    "max_epochs": 200,
    "random_state": RANDOM_STATE,
}

