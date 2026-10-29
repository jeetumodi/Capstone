"""
Preprocessing Architecture module.
Builds a unified ColumnTransformer wrapped inside an sklearn.pipeline.Pipeline:
1. Categorical Pipeline: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
2. Numerical Pipeline: StandardScaler()
All preprocessing is strictly fitted on the training split to prevent data leakage.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """
    Creates the unified ColumnTransformer for concrete parameters.
    - OneHotEncoder applied to categorical variables
    - StandardScaler applied to numerical variables
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(regressor) -> Pipeline:
    """
    Wraps the unified ColumnTransformer and the regressor into an end-to-end Pipeline.
    
    Parameters
    ----------
    regressor : estimator object
        An estimator implementing fit and predict (e.g. Ridge, GaussianNB regressor, XGB, LGBM, ANN).
        
    Returns
    -------
    Pipeline
        sklearn.pipeline.Pipeline combining preprocessing and regression.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", regressor),
        ]
    )

