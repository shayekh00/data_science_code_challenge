from sklearn.impute import KNNImputer
import pandas as pd
from pipeline.transformers.base import BaseTransformer

class KNNImputerWrapper(BaseTransformer):
    def __init__(self, cols_to_impute, n_neighbors=5):
        self.cols_to_impute = cols_to_impute
        self.n_neighbors = n_neighbors
        self.imputer = None
        self.all_numeric_cols = []

    def fit(self, X, y=None):
        X_numeric = X.select_dtypes(include='number')
        self.all_numeric_cols = X_numeric.columns.tolist()

        # Ensure cols_to_impute exist
        missing = [col for col in self.cols_to_impute if col not in self.all_numeric_cols]
        if missing:
            print(f"⚠️ These columns are missing and will be skipped: {missing}")
        self.cols_to_impute = [col for col in self.cols_to_impute if col in self.all_numeric_cols]

        # Fit KNN on numeric data
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(X_numeric)
        return self

    def transform(self, X):
        X = X.copy()
        X_numeric = X.select_dtypes(include='number')

        # Consistency check
        current_cols = X_numeric.columns.tolist()
        if current_cols != self.all_numeric_cols:
            raise ValueError(f"Numeric columns in transform do not match those in fit! Got {current_cols}, expected {self.all_numeric_cols}")
    
        # Impute numeric values
        imputed_array = self.imputer.transform(X_numeric)
        imputed_df = pd.DataFrame(imputed_array, columns=self.all_numeric_cols, index=X.index)

        # Update only the selected columns
        for col in self.cols_to_impute:
            X[col] = imputed_df[col]

        return X
