import pandas as pd
import numpy as np
from sklearn.base import (BaseEstimator, TransformerMixin)

class TotalChargesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if 'customerID' in X.columns:
            X=X.drop(columns='customerID')

        X['TotalCharges']=X['TotalCharges'].astype(str).str.strip().replace(' ', '').replace('', np.nan)

        X['TotalCharges']=pd.to_numeric(X['TotalCharges'], errors= 'coerce')
        
        return X

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.q1_ = X['tenure'].quantile(0.25)
        self.q3_ = X['tenure'].quantile(0.75)

        return self

    def transform(self, X):
        X = X.copy()

        X['tenureMarkdown'] = pd.cut(
            X['tenure'],
            bins=[-np.inf, self.q1_, self.q3_, np.inf],
            labels=[
                'short_term_customer',
                'mid_term_customer',
                'long_term_customer'
            ]
        )

        return X