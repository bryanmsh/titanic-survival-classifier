"""
Kaggle Submission Generation Script.
Fits the locked champion Gradient Boosting pipeline on the entire labeled dataset (train.csv)
and generates Kaggle-formatted predictions for the unlabeled test dataset (test.csv).
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from src.pipeline import build_preprocessor

RANDOM_STATE = 42


def generate_kaggle_submission(
    train_path=os.path.join("data", "train.csv"),
    test_path=os.path.join("data", "test.csv"),
    output_path=os.path.join("data", "submission.csv")
):
    print(f"Loading labeled training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    X_train_full = train_df.drop(columns=['Survived', 'PassengerId'])
    y_train_full = train_df['Survived']

    print(f"Loading unlabeled test data from:     {test_path}")
    test_df = pd.read_csv(test_path)
    passenger_ids = test_df['PassengerId']
    X_test_unlabeled = test_df.drop(columns=['PassengerId'])

    # Build and fit champion pipeline on 100% of labeled training data
    champion_pipeline = Pipeline([
        ('preprocessor', build_preprocessor(scale_numeric=True)),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE
        ))
    ])

    print("Fitting champion pipeline on full training dataset (891 samples)...")
    champion_pipeline.fit(X_train_full, y_train_full)

    print("Generating predictions for held-out Kaggle test partition (418 samples)...")
    predictions = champion_pipeline.predict(X_test_unlabeled)
    probabilities = champion_pipeline.predict_proba(X_test_unlabeled)[:, 1]

    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    # Sanity checks
    assert len(submission_df) == 418, f"Expected 418 rows, got {len(submission_df)}"
    assert submission_df['Survived'].isnull().sum() == 0, "Submission contains null values!"
    assert set(submission_df['Survived'].unique()).issubset({0, 1}), "Invalid class labels found!"

    # Save to disk
    submission_df.to_csv(output_path, index=False)
    print(f"Submission successfully written to: {output_path}")

    # Distribution diagnostics
    pred_survival_rate = submission_df['Survived'].mean() * 100
    train_survival_rate = y_train_full.mean() * 100
    print(f"\n--- Diagnostic Summary ---")
    print(f"Total Test Passengers:           {len(submission_df)}")
    print(f"Predicted Survived:              {(submission_df['Survived'] == 1).sum()} ({pred_survival_rate:.2f}%)")
    print(f"Predicted Deceased:              {(submission_df['Survived'] == 0).sum()} ({100 - pred_survival_rate:.2f}%)")
    print(f"Historical Training Baseline:    {train_survival_rate:.2f}% Survived")
    print(f"Distribution Drift Delta:        {abs(pred_survival_rate - train_survival_rate):.2f}%")

    return submission_df


if __name__ == '__main__':
    generate_kaggle_submission()
