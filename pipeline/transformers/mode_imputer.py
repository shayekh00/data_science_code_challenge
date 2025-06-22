from pipeline.transformers.base import BaseTransformer

class ModeImputer(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.modes_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            if col not in X.columns:
                print(f"⚠️ Column '{col}' not found during fit.")
                continue
            mode_series = X[col].mode()
            if len(mode_series) == 0:
                print(f"⚠️ Column '{col}' has no mode (all missing?)")
            else:
                self.modes_[col] = mode_series.iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col, mode_val in self.modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mode_val)
            else:
                print(f"⚠️ Column '{col}' not found during transform.")
        return X
