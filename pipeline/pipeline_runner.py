# pipeline/pipeline_runner.py
from pipeline.transformers.median_imputer import MedianImputer
from pipeline.transformers.cyclical_encoder import MonthCyclicalEncoder
from pipeline.transformers.outlier import IQRGroupOutlierRemover


class PipelineRunner:
    def __init__(self, steps):
        self.steps = steps  # List of (name, transformer, scope)

    def fit_transform(self, X, y=None):
        for name, transformer, scope in self.steps:
            if scope == 'train' or scope == 'all':
                X = transformer.fit_transform(X, y)
        return X

    def transform(self, X, split='val'):
        for name, transformer, scope in self.steps:
            if scope == 'all':
                X = transformer.transform(X)
        return X
