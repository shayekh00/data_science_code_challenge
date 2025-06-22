# pipeline/transformers/lag_feature_generator.py
import pandas as pd
from pipeline.transformers.base import BaseTransformer

class LagFeatureGenerator(BaseTransformer):
    def __init__(self, col, lags):
        self.col = col
        self.lags = lags
        self.lag_cols_ = []

    def fit(self, X, y=None):
        # No fitting needed — stateless transformer
        return self

    def transform(self, X):
        X = X.copy()
        for lag in self.lags:
            lag_col = f"{self.col}_lag_{lag}"
            X[lag_col] = X[self.col].shift(lag)
            self.lag_cols_.append(lag_col)
        X = X.dropna().reset_index(drop=True)
        return X
