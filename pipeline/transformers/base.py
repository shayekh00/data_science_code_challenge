# pipeline/base.py
class BaseTransformer:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
