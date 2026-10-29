"""
Custom Naive Bayes Regressor Wrapper.
Adapts Gaussian Naive Bayes classification for continuous regression tasks
using quantile discretization and bin-median mapping.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer


class NaiveBayesRegressor(BaseEstimator, RegressorMixin):
    """
    Naive Bayes Regressor using target discretization.
    
    1. In fit(X, y):
       - Discretizes target y into n_bins equal-frequency bins using
         KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile').
       - Trains a standard GaussianNB() model to predict the bin labels.
       - Computes and stores the median y value for each bin.
    2. In predict(X):
       - Predicts the class bin using GaussianNB.
       - Maps each predicted bin to its stored median target value.
    """
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def fit(self, X, y):
        # Convert y to 1D numpy array
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        
        # Discretize continuous target into equal-frequency (quantile) bins
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
        )
        y_binned = self.discretizer_.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(int)
        
        # Train Gaussian Naive Bayes classifier on the bin labels
        self.gnb_ = GaussianNB()
        self.gnb_.fit(X, y_binned)
        
        # Compute and store the median y value for each bin
        self.bin_medians_ = {}
        for b in np.unique(y_binned):
            mask = (y_binned == b)
            self.bin_medians_[int(b)] = float(np.median(y_arr[mask]))
            
        self.global_median_ = float(np.median(y_arr))
        self.classes_ = self.gnb_.classes_
        return self

    def predict(self, X):
        # Predict class bin with GaussianNB
        pred_bins = self.gnb_.predict(X)
        
        # Map each predicted bin to its stored median target value
        mapped_preds = np.array(
            [self.bin_medians_.get(int(b), self.global_median_) for b in pred_bins],
            dtype=np.float64,
        )
        return mapped_preds

