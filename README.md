# Titanic Survival Classifier

**Concentration Pipeline — Project 1: Tabular Machine Learning**  
*An end-to-end, leak-free classical machine learning pipeline predicting passenger survival on the Titanic.*

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

6. **Missing Data Diagnosis & Strategy:**
   - `Age`: 177 missing (19.87%) $\to$ Planned treatment: conditional group median by `Title` and `Pclass`.
   - `Cabin`: 687 missing (77.10%) $\to$ Planned treatment: binary `HasCabin` indicator feature (missingness is informative / MNAR).
   - `Embarked`: 2 missing (0.22%) $\to$ Planned treatment: mode imputation (`'S'`).
   - `Fare`: 1 missing in test set $\to$ Planned treatment: class-based median imputation.

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
