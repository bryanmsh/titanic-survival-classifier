"""
Custom Scikit-Learn Transformers for the Titanic Survival Pipeline.
Designed according to the Scikit-Learn Estimator/Transformer API contract:
Strict separation of parameter estimation (fit) on training data from application (transform).
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts honorific titles from the 'Name' string column.
    Learns common titles during fit (on training data only) and groups infrequent
    titles into a 'Rare' category to prevent high-cardinality noise and category drift.
    """
    def __init__(self, common_titles=('Mr', 'Mrs', 'Miss', 'Master')):
        self.common_titles = list(common_titles)

    def fit(self, X, y=None):
        # Statistically, common titles are predefined or learned from X_train
        return self

    def transform(self, X):
        X_copy = X.copy()
        titles = X_copy['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        X_copy['TitleGroup'] = titles.apply(lambda x: x if x in self.common_titles else 'Rare')
        return X_copy


class FamilyFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    Engineers family dynamics features from 'SibSp' (siblings/spouses) and 'Parch' (parents/children).
    Computes:
    - FamilySize = SibSp + Parch + 1
    - IsAlone = 1 if FamilySize == 1 else 0
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1
        X_copy['IsAlone'] = (X_copy['FamilySize'] == 1).astype(int)
        return X_copy


class CabinIndicator(BaseEstimator, TransformerMixin):
    """
    Extracts a binary presence indicator for 'Cabin'.
    Due to ~77% missingness (predominantly in steerage/3rd class), missingness itself
    is informative (MNAR - Missing Not At Random).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['HasCabin'] = X_copy['Cabin'].notnull().astype(int)
        return X_copy


class GroupedAgeImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'Age' values conditioned on passenger 'Title' and 'Pclass'.
    Learns conditional medians exclusively during `fit` on training data.
    Provides a robust fallback to the training global median for unseen subgroups.
    """
    def __init__(self, common_titles=('Mr', 'Mrs', 'Miss', 'Master')):
        self.common_titles = list(common_titles)
        self.group_medians_ = {}
        self.global_median_ = None

    def fit(self, X, y=None):
        titles = X['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        title_group = titles.apply(lambda x: x if x in self.common_titles else 'Rare')

        temp = pd.DataFrame({
            'TitleGroup': title_group,
            'Pclass': X['Pclass'],
            'Age': X['Age']
        })

        self.global_median_ = float(temp['Age'].median())
        grouped = temp.groupby(['TitleGroup', 'Pclass'])['Age'].median()
        self.group_medians_ = {k: float(v) for k, v in grouped.items() if not pd.isnull(v)}
        return self

    def transform(self, X):
        X_copy = X.copy()
        titles = X_copy['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        title_group = titles.apply(lambda x: x if x in self.common_titles else 'Rare')

        imputed_ages = []
        for (idx, row), tg in zip(X_copy.iterrows(), title_group):
            val = row['Age']
            if pd.isnull(val):
                lookup_key = (tg, row['Pclass'])
                val = self.group_medians_.get(lookup_key, self.global_median_)
            imputed_ages.append(val)

        X_copy['Age'] = imputed_ages
        return X_copy
