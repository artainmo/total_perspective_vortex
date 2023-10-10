from sklearn.base import BaseEstimator, TransformerMixin

class CSPTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y):
        return self

    def transform(self, x, y):
        return x, y
