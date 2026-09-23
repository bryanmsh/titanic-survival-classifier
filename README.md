# Titanic Survival Classifier

*End-to-end, leak-free classical machine learning pipeline predicting passenger survival on the Titanic.*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project implements a disciplined, reproducible machine learning pipeline on tabular data, moving from raw ingestion to model tuning and evaluation. 

The primary pedagogical emphasis is **statistical and engineering rigor**:
- Strict prevention of **data leakage** through Scikit-Learn `Pipeline` and `ColumnTransformer` constructs.
- Establishing naive **baseline classifiers** before advancing to complex models.
- Evaluating multiple model families (**Logistic Regression**, **Random Forest**, **Gradient Boosting**) under identical cross-validation conditions.
- Reasoning about **asymmetric classification errors** (the practical difference between False Positives and False Negatives) rather than relying solely on accuracy.
- Maintaining test set integrity with a single, uncompromised final evaluation.

---

## Dataset Summary

The dataset is derived from the classic **Titanic: Machine Learning from Disaster** challenge:
- **Training Set (`train.csv`):** 891 labeled rows, 12 columns
- **Test Set (`test.csv`):** 418 unlabeled rows, 11 columns (held-out for final inference)
- **Target Variable:** `Survived` ($0 = \text{Perished}, 1 = \text{Survived}$)

### Schema & Features:
| Feature | Type | Description |
|---|---|---|
| `PassengerId` | Identifier | Unique integer ID for each passenger |
| `Survived` | Binary Target | Survival outcome ($0$ or $1$) |
| `Pclass` | Ordinal Categorical | Ticket class ($1 = 1^{\text{st}}, 2 = 2^{\text{nd}}, 3 = 3^{\text{rd}}$) |
| `Name` | Text | Passenger legal name and honorific title |
| `Sex` | Nominal Categorical | Binary biological sex (`male`, `female`) |
| `Age` | Continuous Numeric | Age in years (fractional if $< 1$) |
| `SibSp` | Discrete Numeric | Number of siblings and spouses aboard |
| `Parch` | Discrete Numeric | Number of parents and children aboard |
| `Ticket` | Alphanumeric | Ticket number string |
| `Fare` | Continuous Numeric | Passenger fare in British pounds |
| `Cabin` | Alphanumeric | Cabin number (deck letter and room number) |
| `Embarked` | Nominal Categorical | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |

---

## Exploratory Data Analysis (EDA) Findings

The exploratory analysis is documented in [`titanic_pipeline.ipynb`](./titanic_pipeline.ipynb).

### Key Empirical Findings:

1. **Base Rate & The Accuracy Floor:**
   - Total passengers in training set: 891.
   - Deceased: 549 (61.62%), Survived: 342 (38.38%).
   - A naive `DummyClassifier` predicting the majority class ("all passengers died") achieves **61.62% accuracy** while offering zero predictive utility. Subsequent models must significantly outperform this floor across Precision, Recall, and ROC-AUC.

2. **The Gender Gradient ("Women and Children First"):**
   - **Female Survival Rate:** **74.20%** (233 / 314 survived)
   - **Male Survival Rate:** **18.89%** (109 / 577 survived)
   - Gender represents the single strongest univariate predictor in the entire dataset.

3. **Socioeconomic Class Hierarchy:**
   - **$1^{\text{st}}$ Class:** 62.96% survival
   - **$2^{\text{nd}}$ Class:** 47.28% survival
   - **$3^{\text{rd}}$ Class:** 24.24% survival
   - Passengers in 1st class had privileged proximity to lifeboats and higher social standing during evacuation.

4. **Interaction Effects ($Pclass \times Sex$):**
   - Females in 1st Class (~96.8%) and 2nd Class (~92.1%) had near-certain survival.
   - Females in 3rd Class had survival reduced to 50.0%.
   - Males in 2nd and 3rd Class faced severe mortality (15.7% and 13.5% survival, respectively).

5. **Family Dynamics:**
   - Solo travelers (`SibSp + Parch == 0`) had a survival rate of ~30.4%.
   - Moderate families (2–4 total members) had survival rates above 55%.
   - Very large families (5+ members) suffered severe mortality ($< 20\%$), likely due to evacuation chaos.

---

## Data Partitioning & Leak-Free Architecture

To prevent data leakage, the training dataset is partitioned and preprocessed using a strictly decoupled architecture:

1. **Stratified Partitioning:**
   - Partitioned into **80% Development Training** (712 samples) and **20% Held-Out Test** (179 samples).
   - Stratified on `Survived` to ensure identical target distributions across both partitions (38.34% vs. 38.55%).
   - The 20% test partition remains completely isolated and untouched until final evaluation.

2. **Custom Scikit-Learn Transformers (`src/transformers.py`):**
   - `TitleExtractor`: Extracts honorific titles from `Name` and collapses rare titles into an `'Rare'` bucket.
   - `FamilyFeaturesAdder`: Computes `FamilySize` and `IsAlone`.
   - `CabinIndicator`: Extracts binary presence flag `HasCabin`.
   - `GroupedAgeImputer`: Imputes missing `Age` values using conditional group medians (`Title` $\times$ `Pclass`) fitted **exclusively on the training split**.

---

## Feature Engineering & Preprocessing Pipeline

All feature transformations, conditional imputation, encoding, and scaling are unified into a composite Scikit-Learn `ColumnTransformer` and `Pipeline` ([`src/pipeline.py`](./src/pipeline.py)):

1. **Feature Engineering Chain (`FeatureEngineer`):**
   - Applies `TitleExtractor`, `FamilyFeaturesAdder`, `CabinIndicator`, and `GroupedAgeImputer` in strict sequence.
2. **Parallel Column Routing (`ColumnTransformer`):**
   - **Numeric Branch (`['Age', 'Fare', 'FamilySize', 'Pclass']`):** Imputed via median and standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
   - **Categorical Branch (`['Sex', 'Embarked', 'TitleGroup']`):** Mode-imputed and one-hot encoded with `handle_unknown='ignore'`.
   - **Binary Branch (`['IsAlone', 'HasCabin']`):** Passed through as clean binary indicators.
   - **Dropped Features:** Raw text fields and components (`PassengerId`, `Ticket`, `Cabin`, `Name`, `SibSp`, `Parch`).
3. **Design Matrix Output:**
   - Produces a leak-free 16-feature design matrix ready for model training, with zero missing values across both train and held-out test partitions.

---

## Model Exploration & Cross-Validation Results

We evaluate three diverse model families alongside a naive baseline using **Stratified 5-Fold Cross-Validation** strictly on the training partition (`X_train` = 712 samples). Preprocessing pipelines are re-fit inside each fold to guarantee zero data leakage ([`src/models.py`](./src/models.py)).

### Cross-Validation Performance Summary:

| Model Family | Accuracy (Mean $\pm$ Std) | Precision | Recall | $F_1$-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Dummy (Baseline)** | $61.66\% \pm 0.27\%$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.5000$ |
| **Logistic Regression** | **$83.43\% \pm 2.37\%$** | $0.7981$ | **$0.7657$** | **$0.7806$** | $0.8716$ |
| **Random Forest** | $81.60\% \pm 2.17\%$ | $0.7704$ | $0.7436$ | $0.7563$ | $0.8759$ |
| **Gradient Boosting** | $81.60\% \pm 3.13\%$ | **$0.7948$** | $0.7107$ | $0.7480$ | **$0.8922$** |

### Key Evaluation Takeaways:
1. **The Floor & Accuracy Paradox:** `DummyClassifier` achieves $61.66\%$ accuracy by predicting that all passengers perished, while offering zero practical utility ($F_1 = 0.0000$, $\text{ROC-AUC} = 0.5000$).
2. **Linear Parametric Baseline:** `LogisticRegression` delivers strong out-of-fold performance ($83.43\%$ Accuracy, $0.7806$ $F_1$), confirming the signal unlocked by our feature engineering.
3. **Threshold Discrimination:** `GradientBoostingClassifier` achieved the highest discrimination power (**$\text{ROC-AUC} = 0.8922$**), excelling at probabilistic ranking across decision thresholds.

---

## Hyperparameter Optimization & Model Selection

Following initial cross-validation, the highest-ranking candidate model (**`GradientBoostingClassifier`**) was optimized using `GridSearchCV` with Stratified 5-Fold Cross-Validation ([`src/models.py`](./src/models.py)).

### Parameter Search Space & Tuning Strategy:
To guard against validation overfitting on a modest dataset (~700 training observations), the search space was bounded around structural tree complexity and regularization controls:
- `n_estimators`: `[80, 100, 120]`
- `learning_rate`: `[0.03, 0.05, 0.1]`
- `max_depth`: `[2, 3]`
- `subsample`: `[0.8, 1.0]`

### Optimization Results:

| Hyperparameter | Selected Value | Structural Justification |
|---|---|---|
| `learning_rate` | `0.1` | Optimal convergence without over-iterating on residual noise |
| `max_depth` | `3` | Captures 3-way interactions (`Sex` $\times$ `Pclass` $\times$ `AgeGroup`) while preventing leaf memorization |
| `n_estimators` | `100` | Sufficient ensemble capacity prior to validation score plateau |
| `subsample` | `0.8` | Injects stochastic bagging randomness to reduce correlation across sequential trees |

**Outcome:** Tuned cross-validated **ROC-AUC reached 0.8938** (improving over the untuned baseline of 0.8922), establishing our champion model.

---

## Final Held-Out Test Set Evaluation

Following strict experimental protocols to prevent **Data Snooping Bias**, the 20% test partition (`X_test`, `y_test`: 179 samples) established during Phase 2 was kept completely unexamined and untouched until model development was frozen.

The tuned champion pipeline was fit on the full 80% training set and evaluated **exactly once** on the held-out test partition:

| Metric | Held-Out Test (20%) | 5-Fold CV Mean | Generalization Delta |
|---|---|---|---|
| **Accuracy** | **79.89%** | $82.87\%$ | $-2.98\%$ |
| **Precision** | **76.19%** | $79.56\%$ | $-3.37\%$ |
| **Recall** | **69.57%** | $73.65\%$ | $-4.08\%$ |
| **$F_1$-Score** | **72.73%** | $76.24\%$ | $-3.51\%$ |
| **ROC-AUC** | **0.8343** | $0.8938$ | $-0.0595$ |

### Held-Out Test Confusion Matrix:
$$\begin{pmatrix} \text{True Negatives (TN)} = 95 & \text{False Positives (FP)} = 15 \\ \text{False Negatives (FN)} = 21 & \text{True Positives (TP)} = 48 \end{pmatrix}$$

- **Zero Data Snooping:** Test metrics closely align with cross-validation expectations without catastrophic distribution drop-off.
- **Asymmetric Error Profile:** With only 15 False Positives and 21 False Negatives out of 179 samples, the model maintains a well-calibrated decision boundary. In life-critical rescue allocations, the discrimination threshold $\tau$ can be tuned downward from $0.5$ to increase Recall and minimize False Negatives.

---

## Model Interpretability & Feature Importance Diagnostics

We evaluate feature importance using two complementary lenses: **Mean Decrease in Impurity (MDI / Gini Importance)** and out-of-sample **Permutation Feature Importance (PFI)** on the held-out test distribution.

| Rank | Transformed Feature | MDI (Gini Importance) | Permutation Importance ($\Delta\text{ROC-AUC}$) | Domain Mechanism |
|:---:|---|:---:|:---:|---|
| **1** | `TitleGroup_Mr` | **0.285** | **0.108** | Adult male status; primary factor in "women and children first" |
| **2** | `Pclass` | 0.110 | **0.044** | Socioeconomic status; proximity and prioritized access to lifeboats |
| **3** | `Fare` | **0.204** | 0.026 | Correlated with upper decks (high MDI score driven by cardinality bias) |
| **4** | `Age` | 0.121 | 0.022 | Priority evacuation of children vs. elderly adults |
| **5** | `FamilySize` | 0.041 | 0.019 | Moderate families (2–4) survived best; solo and large families struggled |
| **6** | `Sex_female` | 0.094 | 0.015 | Direct gender confirmation across titles |

### Key Diagnostic Insight (MDI vs. Permutation Importance):
MDI places continuous `Fare` at #2 ($0.204$) ahead of ordinal `Pclass` ($0.110$) because continuous variables have many candidate split thresholds. However, Permutation Feature Importance on held-out test data reveals that `Pclass` is actually substantially more influential ($\Delta\text{AUC} = 0.044$) than granular `Fare` variations ($\Delta\text{AUC} = 0.026$). This empirically validates the necessity of permutation testing to correct for MDI cardinality bias.

---

## Comprehensive Multi-Model Benchmark

| Model Architecture | Evaluation Protocol | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline (Majority Class)** | 5-Fold Stratified CV | 61.66% | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Logistic Regression (L2 Regularized)** | 5-Fold Stratified CV | 83.43% | 0.7981 | 0.7657 | 0.7806 | 0.8716 |
| **Random Forest (100 Trees)** | 5-Fold Stratified CV | 81.60% | 0.7704 | 0.7436 | 0.7563 | 0.8759 |
| **Gradient Boosting (Default / Untuned)** | 5-Fold Stratified CV | 81.60% | 0.7948 | 0.7107 | 0.7480 | 0.8922 |
| **Gradient Boosting (Tuned Champion)** | 5-Fold Stratified CV | 82.87% | 0.7956 | 0.7365 | 0.7624 | **0.8938** |
| **Gradient Boosting (Tuned Champion)** | **Held-Out Test Set (20%)** | **79.89%** | **0.7619** | **0.6957** | **0.7273** | **0.8343** |

---

## Key Engineering Takeaways

1. **Leak-Free Discipline:** Feature engineering, conditional imputation (e.g. median age by `Title` $\times$ `Pclass`), and scaling must be encapsulated in reusable Scikit-Learn transformers and nested inside cross-validation loops to prevent optimistic performance bias.
2. **Beyond Accuracy:** Trivial baseline models can achieve $>60\%$ accuracy on unbalanced problems. Proper evaluation requires Precision, Recall, $F_1$, and threshold-independent ROC-AUC coupled with domain error cost analysis.
3. **Tree Regularization:** Bounding max tree depth (`max_depth = 3`) and applying stochastic subsampling (`subsample = 0.8`) protects boosting ensembles from overfitting modest tabular datasets.

---

## Local Setup & Reproduction

To reproduce this project locally:

```bash
# 1. Clone the repository
git clone https://github.com/bryanmsh/titanic-survival-classifier.git
cd titanic-survival-classifier

# 2. Create and activate a Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the Pipeline Notebook
jupyter notebook titanic_pipeline.ipynb
```

