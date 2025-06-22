# pipeline/transformers/cyclical_encoder.py
import numpy as np
from .base import BaseTransformer

class MonthCyclicalEncoder(BaseTransformer):
    def transform(self, X):
        X = X.copy()
        X['month'] = X['date'].dt.month
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        
        return X.drop(columns=['month'])
