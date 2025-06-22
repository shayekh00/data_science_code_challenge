from pipeline.transformers.base import BaseTransformer
import numpy as np

class FeatureDropper(BaseTransformer):
    def __init__(self, always_drop=None, missing_thresh=0.95, corr_thresh=0.90):
        self.always_drop = always_drop or []
        self.missing_thresh = missing_thresh
        self.corr_thresh = corr_thresh
        self.drop_from_train = []

    def fit(self, X, y=None):
        X = X.copy()

        # 1. Drop high-missing columns (> missing_thresh)
        missing_frac = X.isna().mean()
        drop_missing = missing_frac[missing_frac > self.missing_thresh].index.tolist()

        # 2. Drop highly correlated columns
        numeric = X.select_dtypes(include='number')
        corr_matrix = numeric.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        drop_corr = [col for col in upper.columns if any(upper[col] > self.corr_thresh)]

        # Save columns to drop from training logic
        self.drop_from_train = list(set(drop_missing + drop_corr))

        print(f"FeatureDropper fit complete. {len(X.columns)} total columns.")
        print(f"Dropping {len(self.always_drop)} always dropped columns: {self.always_drop}")
        print(f"Dropping {len(drop_missing)} columns due to high missing values: {drop_missing}")
        print(f"Dropping {len(drop_corr)} columns due to high correlation: {drop_corr}")
        print(f"Dropping {len(self.drop_from_train)} columns from training: {self.drop_from_train}")
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = self.always_drop + self.drop_from_train
        existing = [col for col in drop_cols if col in X.columns]
        return X.drop(columns=existing)
