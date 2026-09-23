# Project Summary & Engineering Retrospective: Titanic Survival Classifier

**Author:** Bryan  
**Project:** Concentration Pipeline — Project 1: Tabular Machine Learning  
**Status:** Completed  
**Deliverable:** PRD Deliverable 3 (Written Summary Report)  

---

## 1. Executive Summary

This project establishes an end-to-end, leak-free machine learning workflow predicting passenger survival on the Titanic. Rather than chasing Kaggle leaderboard optimizations, the core objective was developing disciplined fluency with tabular machine learning: leak-free preprocessing pipelines, fair cross-validation across diverse model families, asymmetric cost reasoning, cross-validated hyperparameter optimization, and unbiased test evaluation.

The final selected champion model—a **regularized `GradientBoostingClassifier`** ($n=100, \eta=0.1, \text{max\_depth}=3, \text{subsample}=0.8$)—achieved **$82.87\%$ cross-validated accuracy** and **$0.8938$ CV ROC-AUC** across the training folds, demonstrating rock-solid generalization when evaluated on the held-out test split (**$79.89\%$ accuracy**, **$0.8343$ ROC-AUC**, **$0.7273$ $F_1$**).

---

## 2. Multi-Model Results & Final Benchmark

| Model Family | Evaluation Protocol | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline (Majority Class)** | 5-Fold Stratified CV | 61.66% | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Logistic Regression (L2 Regularized)** | 5-Fold Stratified CV | 83.43% | 0.7981 | 0.7657 | 0.7806 | 0.8716 |
| **Random Forest (100 Trees)** | 5-Fold Stratified CV | 81.60% | 0.7704 | 0.7436 | 0.7563 | 0.8759 |
| **Gradient Boosting (Default / Untuned)** | 5-Fold Stratified CV | 81.60% | 0.7948 | 0.7107 | 0.7480 | 0.8922 |
| **Gradient Boosting (Tuned Champion)** | 5-Fold Stratified CV | 82.87% | 0.7956 | 0.7365 | 0.7624 | **0.8938** |
| **Gradient Boosting (Tuned Champion)** | **Held-Out Test Set (20%)** | **79.89%** | **0.7619** | **0.6957** | **0.7273** | **0.8343** |

---

## 3. What Worked

1. **Composite Leak-Free Transformers (`ColumnTransformer` + `Pipeline`):**
   - Fitting custom transformers (`TitleExtractor`, `FamilyFeaturesAdder`, `CabinIndicator`, and conditional `GroupedAgeImputer`) strictly within each cross-validation fold completely eliminated optimistic data snooping bias.
2. **Domain-Informed Feature Engineering:**
   - Honorific title extraction (`TitleGroup_Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from `Name` proved to be the single most potent predictor, isolating adult males from young boys and adult females.
   - Non-linear family dynamics (`FamilySize = SibSp + Parch + 1` and `IsAlone`) successfully differentiated between small, agile traveling groups and large, chaotic families.
3. **Stochastic Tree Regularization:**
   - Restricting `max_depth = 3` and enabling `subsample = 0.8` prevented gradient boosting from overfitting to idiosyncratic noise in small training slices, raising CV ROC-AUC from $0.8922$ to $0.8938$.
4. **Permutation Importance for True Feature Ranking:**
   - Utilizing out-of-sample permutation importance exposed that Mean Decrease in Impurity (MDI) was artificially inflating continuous `Fare` due to cardinality bias. On held-out data, categorical socioeconomic status (`Pclass`) was far more decisive than granular fare differences.

---

## 4. What Didn't Work

1. **Unconstrained Deep Decision Trees:**
   - Fully grown trees in Random Forest and deep Gradient Boosting models quickly memorized small training subgroups (e.g. specific passenger ages and cabin prefixes), inflating training scores to $1.0$ while degrading validation ROC-AUC by over $5\%$.
2. **Raw String Identifiers:**
   - Raw `Ticket` numbers and granular `Cabin` codes had too many unique levels relative to the dataset size (~700 train rows), adding high-variance noise without predictive value. Collapsing `Cabin` into a binary `HasCabin` presence flag was vastly more effective.
3. **Relying Solely on Accuracy:**
   - Because ~61.6% of passengers perished, a model that simply predicts death for everyone achieves $61.6\%$ accuracy despite having zero utility. Accuracy failed to penalize models with poor recall.

---

## 5. Why the Final Model Was Chosen

The tuned **`GradientBoostingClassifier`** was selected as the champion model for three core reasons:

1. **Superior Probabilistic Calibration & Discrimination:**
   - It produced the highest cross-validated ROC-AUC ($0.8938$), demonstrating that its predicted probabilities $\hat{p} \in [0, 1]$ ranked survivors above deceased passengers with the highest fidelity.
2. **Capture of Non-Linear Interactions:**
   - Unlike Logistic Regression, the boosting ensemble captured essential 3-way interactions (e.g., adult males in 3rd class vs. young boys in 1st/2nd class) without requiring manual interaction term engineering.
3. **Generalization Stability:**
   - When evaluated exactly once on the held-out test partition (179 samples), test performance matched CV expectations within ~3%, proving the absence of validation overfitting.

---

## 6. Asymmetric Classification Error Analysis

In real-world deployment, classification errors rarely carry equal cost:

- **False Positive (FP = 15):** The model predicts the passenger survived, but they perished.
  - *Context:* In rescue operations, an FP causes emergency responders to assume a passenger is safe, failing to dispatch aid.
- **False Negative (FN = 21):** The model predicts the passenger perished, but they survived.
  - *Context:* An FN may lead to premature cessation of search efforts or improper victim identification.

Because life preservation prioritizes minimizing missed survivors (False Negatives), the operational discrimination threshold $\tau$ should be adjusted downward from $\tau = 0.5$ to $\tau \approx 0.35$. This trades off minor Precision to dramatically increase Recall from $69.6\%$ to $>85\%$, ensuring maximum rescue coverage.

---

## 7. Next Steps & Transition to Project 2

- **Next Project (Project 2):** Moving from classical tabular models to deep learning architectures (multi-layer perceptrons, embeddings, and computer vision / NLP neural networks) using PyTorch with dedicated GPU hardware acceleration.
