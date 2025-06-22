import pandas as pd
from pipeline.transformers.base import BaseTransformer

class TargetLagTransformer(BaseTransformer):
    def __init__(self, target_col='snow', lag_target=True, lag_features=None, rolling_features=None, groupby_col=None):
        """
        Parameters:
        - target_col: str — the name of the target column to shift
        - lag_target: bool — whether to create `target_col + '_tomorrow'`
        - lag_features: dict — {column_name: list_of_lags}, e.g., {'mean_temp': [1, 2]}
        - rolling_features: dict — {column_name: list_of_window_sizes}, e.g., {'total_precipitation': [3]}
        - groupby_col: str — optional, for station-wise lagging/rolling
        """
        self.target_col = target_col
        self.lag_target = lag_target
        self.lag_features = lag_features or {}
        self.rolling_features = rolling_features or {}
        self.groupby_col = groupby_col

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        X = X.copy()
        if self.groupby_col:
            groups = X.groupby(self.groupby_col, group_keys=False)
        else:
            groups = [(None, X)]

        dfs = []
        for _, group in groups:
            df = group.copy()

            if self.lag_target:
                df[f'{self.target_col}_tomorrow'] = df[self.target_col].shift(-1)

            for col, lags in self.lag_features.items():
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)

            for col, windows in self.rolling_features.items():
                for w in windows:
                    df[f'{col}_roll{w}'] = df[col].rolling(w).sum().shift(1)

            dfs.append(df)

        X_new = pd.concat(dfs, axis=0).sort_index()

        # Drop rows with NaNs introduced by shifting
        X_new = X_new.dropna(subset=[f'{self.target_col}_tomorrow'] if self.lag_target else [])

        return X_new
