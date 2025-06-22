from pipeline.transformers.base import BaseTransformer

class BinaryFlagImputerEncoder(BaseTransformer):
    def __init__(self, flag_cols):
        self.flag_cols = flag_cols

    def fit(self, X, y=None):
        # Nothing to learn from data
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.flag_cols:
            if col not in X.columns:
                continue
            # Fill missing with False
            X[col] = X[col].fillna(False)

            # Coerce to boolean first, then int
            X[col] = X[col].astype(bool).astype(int)
        return X
