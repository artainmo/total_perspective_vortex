from sklearn.base import BaseEstimator, TransformerMixin

class CSPTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x):
        return self

    def transform(self, x):
        return x
