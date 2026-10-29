"""
Configuration and Schema Definitions for Model Comparison and ACI Constraint Auditing.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "4thsem" / "Data.csv"
SAVED_MODELS_4THSEM = ROOT_DIR / "4thsem" / "saved_models"
PIM_MTL_DIR = ROOT_DIR / "3rdYearsem5"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Feature Schema
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

# Targets
TARGET_VARIABLES = {
    "Cs": "Cs_Mpa",
    "Ts": "Ts_Mpa",
    "Fs": "Fs_Mpa",
}

# 30% Test Split Configuration
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ACI 318-19 & Eurocode 2 Physical Ratio Bounds
ACI_BOUNDS = {
    "Rt_min": 0.06,  # Min Tensile Ratio: Ts / Cs >= 0.06
    "Rt_max": 0.18,  # Max Tensile Ratio: Ts / Cs <= 0.18
    "Rf_min": 0.08,  # Min Flexural Ratio: Fs / Cs >= 0.08
    "Rf_max": 0.25,  # Max Flexural Ratio: Fs / Cs <= 0.25
}

# Model Identifiers and Display Labels
MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "naive_bayes": "Naive Bayes Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
    "ann": "Single-Task ANN (PyTorch)",
    "pim_mtl": "PIM-MTL",
}

# Color palette for consistent, publication-ready figures
MODEL_COLORS = {
    "Ridge Regression": "#6c757d",           # Grey
    "Naive Bayes Regressor": "#ffc107",      # Amber
    "XGBoost Regressor": "#17a2b8",          # Cyan / Teal
    "LightGBM Regressor": "#007bff",         # Blue
    "Single-Task ANN (PyTorch)": "#fd7e14",   # Orange
    "PIM-MTL": "#28a745",                    # Green
}
