# pipeline/transformers/outlier.py
from .base import BaseTransformer
import numpy as np

class IQRGroupOutlierRemover(BaseTransformer):
    def __init__(self, group_col, cols=None, multiplier=1.5):
        self.group_col = group_col
        self.cols = cols
        self.multiplier = multiplier
        self.thresholds_ = {}

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols_ = X.select_dtypes(include='number').columns.tolist()
            if self.group_col in self.cols_:
                self.cols_.remove(self.group_col)
        else:
            self.cols_ = self.cols

        self.thresholds_ = {}
        grouped = X.groupby(self.group_col)
        for group_name, group_df in grouped:
            self.thresholds_[group_name] = {}
            for col in self.cols_:
                Q1 = group_df[col].quantile(0.25)
                Q3 = group_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.multiplier * IQR
                upper = Q3 + self.multiplier * IQR
                self.thresholds_[group_name][col] = (lower, upper)
        return self

    def transform(self, X):
        def filter_group(group):
            group_name = group[self.group_col].iloc[0]
            if group_name not in self.thresholds_:
                return group  # fallback: don't filter this group

            for col in self.cols_:
                lower, upper = self.thresholds_[group_name].get(col, (-np.inf, np.inf))
                group = group[(group[col] >= lower) & (group[col] <= upper)]
            return group

        return X.groupby(self.group_col, group_keys=False).apply(filter_group)



# class IQRGroupOutlierRemover(BaseTransformer):
#     def __init__(self, group_col, cols=None, multiplier=1.5):
#         self.group_col = group_col
#         self.cols = cols
#         self.multiplier = multiplier

#     def fit(self, X, y=None):
#         if self.cols is None:
#             self.cols_ = X.select_dtypes(include='number').columns.tolist()
#             if self.group_col in self.cols_:
#                 self.cols_.remove(self.group_col)
#         else:
#             self.cols_ = self.cols
#         return self

#     def transform(self, X):
#         def filter_group(group):
#             for col in self.cols_:
#                 Q1 = group[col].quantile(0.25)
#                 Q3 = group[col].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower = Q1 - self.multiplier * IQR
#                 upper = Q3 + self.multiplier * IQR
#                 group = group[(group[col] >= lower) & (group[col] <= upper)]
#             return group
#         return X.groupby(self.group_col, group_keys=False).apply(filter_group)