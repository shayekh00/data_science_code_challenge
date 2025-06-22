from pipeline.transformers.base import BaseTransformer

class MeanImputer(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.means_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                self.means_[col] = X[col].mean()
            else:
                print(f"⚠️ Column '{col}' not found during fit.")
        return self

    def transform(self, X):
        X = X.copy()
        for col, mean_val in self.means_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mean_val)
            else:
                print(f"⚠️ Column '{col}' not found during transform.")
        return X
