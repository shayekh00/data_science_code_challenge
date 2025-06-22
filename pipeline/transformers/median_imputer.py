from .base import BaseTransformer

class MedianImputer(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.medians = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.medians[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.medians[col])
        return X
