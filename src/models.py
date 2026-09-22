"""
Model Configuration & Cross-Validation Evaluation Module.
Implements candidate classifiers and leak-free stratified cross-validation routines.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from src.pipeline import build_preprocessor


def get_candidate_models(random_state=42):
    """
    Returns a dictionary of instantiated candidate classifiers:
    1. Baseline: DummyClassifier (most frequent class)
    2. Linear Model: LogisticRegression (L2 regularized)
    3. Bagging Ensemble: RandomForestClassifier
    4. Boosting Ensemble: GradientBoostingClassifier
    """
    return {
        'Dummy (Baseline)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    }


def evaluate_models_cv(X, y, cv_splits=5, random_state=42):
    """
    Evaluates candidate models using Stratified K-Fold Cross-Validation.
    Each model is chained with the full ColumnTransformer preprocessor inside a Pipeline,
    ensuring all transformations are fit strictly on training folds with zero leakage.

    Returns:
        tuple: (results_df, oof_predictions_dict, oof_probas_dict)
    """
    models = get_candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    records = []
    oof_predictions = {}
    oof_probabilities = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', build_preprocessor(scale_numeric=True)),
            ('classifier', model)
        ])

        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)

        records.append({
            'Model': name,
            'Accuracy (Mean)': scores['test_accuracy'].mean(),
            'Accuracy (Std)': scores['test_accuracy'].std(),
            'Precision': scores['test_precision'].mean(),
            'Recall': scores['test_recall'].mean(),
            'F1-Score': scores['test_f1'].mean(),
            'ROC-AUC': scores['test_roc_auc'].mean(),
        })

        # Out-of-fold predictions for confusion matrix and ROC curves
        oof_preds = cross_val_predict(pipe, X, y, cv=cv, method='predict')
        oof_predictions[name] = oof_preds

        if hasattr(model, 'predict_proba'):
            oof_probs = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
        else:
            oof_probs = np.zeros(len(y))
        oof_probabilities[name] = oof_probs

    results_df = pd.DataFrame(records).set_index('Model')
    return results_df, oof_predictions, oof_probabilities
