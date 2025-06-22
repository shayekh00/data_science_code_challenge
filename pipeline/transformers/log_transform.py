import numpy as np
from pipeline.transformers.base import BaseTransformer

class LogTransformer(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.valid_cols_ = []

    def fit(self, X, y=None):
        # Validate columns
        self.valid_cols_ = []
        for col in self.cols:
            if col not in X.columns:
                print(f"⚠️ Column '{col}' not found — skipping log transform.")
                continue
            if not np.issubdtype(X[col].dtype, np.number):
                print(f"⚠️ Column '{col}' is not numeric — skipping log transform.")
                continue
            if (X[col] < 0).any():
                print(f"⚠️ Column '{col}' contains negative values — skipping log transform.")
                continue
            self.valid_cols_.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.valid_cols_:
            X[col] = np.log1p(X[col])
        return X
