"""
Model Training and 5-Fold Cross-Validation Hyperparameter Tuning Module.
Implements:
1. Ridge Regression (tuning alpha)
2. Custom Naive Bayes Regressor (20 quantile bins)
3. XGBoost Regressor (tuning max_depth, learning_rate, subsample, colsample_bytree)
4. LightGBM Regressor (tuning num_leaves, learning_rate, subsample, colsample_bytree)
5. PyTorch ANN Regressor (tuning H and dropout_p with CosineAnnealingLR & Early Stopping)
All preprocessing is strictly contained in pipelines and fitted only on training folds.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    RIDGE_PARAM_GRID,
    NAIVE_BAYES_N_BINS,
    XGB_PARAM_GRID,
    XGB_FIXED_PARAMS,
    LGBM_PARAM_GRID,
    LGBM_FIXED_PARAMS,
    ANN_GRID,
    ANN_TRAINING_CONFIG,
)
from pipeline import build_pipeline, create_preprocessor
from models.naive_bayes_regressor import NaiveBayesRegressor
from models.ann import PyTorchANNRegressor


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune and train Ridge Regression with 5-Fold CV."""
    print("  -> Tuning Ridge Regression (alpha in [0.01, 0.1, 1.0, 10.0, 100.0])...")
    base_pipe = build_pipeline(Ridge(random_state=RANDOM_STATE))
    search = GridSearchCV(
        estimator=base_pipe,
        param_grid=RIDGE_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)
    print(f"     Best Ridge CV R2: {best_score:.4f} | Params: {best_params}")
    return best_pipe, best_params, best_score


def train_naive_bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Dict[str, Any], float]:
    """Train Naive Bayes Regressor and evaluate with 5-Fold CV."""
    print(f"  -> Training Naive Bayes Regressor ({NAIVE_BAYES_N_BINS} quantile bins)...")
    nb_pipe = build_pipeline(NaiveBayesRegressor(n_bins=NAIVE_BAYES_N_BINS))
    cv_scores = cross_val_score(
        nb_pipe,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    mean_cv = float(np.mean(cv_scores))
    nb_pipe.fit(X_train, y_train)
    params = {"n_bins": NAIVE_BAYES_N_BINS}
    print(f"     Naive Bayes Mean CV R2: {mean_cv:.4f}")
    return nb_pipe, params, mean_cv


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune and train XGBoost Regressor with 5-Fold CV."""
    print("  -> Tuning XGBoost Regressor (depth, lr, subsample, colsample)...")
    xgb_reg = xgb.XGBRegressor(**XGB_FIXED_PARAMS)
    base_pipe = build_pipeline(xgb_reg)
    search = GridSearchCV(
        estimator=base_pipe,
        param_grid=XGB_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)
    print(f"     Best XGBoost CV R2: {best_score:.4f} | Params: {best_params}")
    return best_pipe, best_params, best_score


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tune and train LightGBM Regressor with 5-Fold CV."""
    print("  -> Tuning LightGBM Regressor (num_leaves, lr, subsample, colsample)...")
    lgb_reg = lgb.LGBMRegressor(**LGBM_FIXED_PARAMS)
    base_pipe = build_pipeline(lgb_reg)
    search = GridSearchCV(
        estimator=base_pipe,
        param_grid=LGBM_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=2,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)
    print(f"     Best LightGBM CV R2: {best_score:.4f} | Params: {best_params}")
    return best_pipe, best_params, best_score


def train_ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune PyTorch ANN Regressor over H in {128, 256} and dropout_p in {0.2, 0.3}
    using 5-Fold Cross-Validation, then refit the best configuration on full training data.
    """
    print("  -> Tuning PyTorch ANN (H in {128, 256}, dropout_p in {0.2, 0.3}) with 5-Fold CV...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    best_cv_score = -float("inf")
    best_cfg = None

    for cfg in ANN_GRID:
        h_val = cfg["H"]
        p_val = cfg["dropout_p"]
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            fold_preprocessor = create_preprocessor()
            X_tr_fold = fold_preprocessor.fit_transform(X_train.iloc[train_idx])
            X_va_fold = fold_preprocessor.transform(X_train.iloc[val_idx])
            y_tr_fold = y_train.iloc[train_idx].values
            y_va_fold = y_train.iloc[val_idx].values

            ann_model = PyTorchANNRegressor(
                H=h_val,
                dropout_p=p_val,
                lr=ANN_TRAINING_CONFIG["lr"],
                weight_decay=ANN_TRAINING_CONFIG["weight_decay"],
                batch_size=ANN_TRAINING_CONFIG["batch_size"],
                max_epochs=ANN_TRAINING_CONFIG["max_epochs"],
                patience=ANN_TRAINING_CONFIG["patience"],
                random_state=RANDOM_STATE + fold_idx,
            )
            ann_model.fit(X_tr_fold, y_tr_fold)
            preds = ann_model.predict(X_va_fold)
            score = float(r2_score(y_va_fold, preds))
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        print(f"     ANN Config H={h_val}, p={p_val} -> Mean CV R2: {mean_score:.4f}")
        
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_cfg = cfg

    print(f"     Best ANN CV R2: {best_cv_score:.4f} | Config: {best_cfg}")

    # Refit best configuration on the entire 70% training split inside full Pipeline
    best_ann = PyTorchANNRegressor(
        H=best_cfg["H"],
        dropout_p=best_cfg["dropout_p"],
        lr=ANN_TRAINING_CONFIG["lr"],
        weight_decay=ANN_TRAINING_CONFIG["weight_decay"],
        batch_size=ANN_TRAINING_CONFIG["batch_size"],
        max_epochs=ANN_TRAINING_CONFIG["max_epochs"],
        patience=ANN_TRAINING_CONFIG["patience"],
        random_state=RANDOM_STATE,
    )
    final_pipeline = build_pipeline(best_ann)
    final_pipeline.fit(X_train, y_train)

    return final_pipeline, best_cfg, best_cv_score

