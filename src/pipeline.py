"""
Pipeline Construction Module for Titanic Survival Classifier.
Encapsulates feature engineering, imputation, scaling, and categorical encoding
into a single, unified Scikit-Learn Pipeline and ColumnTransformer.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

from src.transformers import (
    TitleExtractor,
    FamilyFeaturesAdder,
    CabinIndicator,
    GroupedAgeImputer
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies custom feature engineering transformations in strict sequence:
    1. Title extraction from 'Name'
    2. Family dynamics ('FamilySize', 'IsAlone') from 'SibSp' and 'Parch'
    3. 'HasCabin' presence flag from 'Cabin'
    4. Grouped conditional 'Age' imputation (fit exclusively on training split)
    """
    def __init__(self, common_titles=('Mr', 'Mrs', 'Miss', 'Master')):
        self.common_titles = common_titles
        self.title_extractor = TitleExtractor(common_titles=self.common_titles)
        self.family_adder = FamilyFeaturesAdder()
        self.cabin_indicator = CabinIndicator()
        self.age_imputer = GroupedAgeImputer(common_titles=self.common_titles)

    def fit(self, X, y=None):
        X_t = X.copy()
        X_t = self.title_extractor.fit_transform(X_t)
        X_t = self.family_adder.fit_transform(X_t)
        X_t = self.cabin_indicator.fit_transform(X_t)
        self.age_imputer.fit(X_t)
        return self

    def transform(self, X):
        X_t = X.copy()
        X_t = self.title_extractor.transform(X_t)
        X_t = self.family_adder.transform(X_t)
        X_t = self.cabin_indicator.transform(X_t)
        X_t = self.age_imputer.transform(X_t)
        return X_t


def build_preprocessor(scale_numeric=True):
    """
    Constructs an end-to-end ColumnTransformer preprocessing pipeline.

    Args:
        scale_numeric (bool): Whether to apply StandardScaler to numeric features.
                             Defaults to True (required for Logistic Regression,
                             safe/invariant for tree ensembles).

    Returns:
        Pipeline: Composite Scikit-Learn Pipeline ready for fit/transform.
    """
    # 1. Numeric Feature Pipeline
    numeric_features = ['Age', 'Fare', 'FamilySize', 'Pclass']
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # 2. Categorical Feature Pipeline
    categorical_features = ['Sex', 'Embarked', 'TitleGroup']
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Binary Pass-Through Features
    binary_features = ['IsAlone', 'HasCabin']

    # 4. Assemble ColumnTransformer
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
            ('bin', 'passthrough', binary_features),
        ],
        remainder='drop'
    )

    # 5. Chain Feature Engineering with ColumnTransformer
    full_preprocessor = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('column_transform', col_transformer)
    ])

    return full_preprocessor
