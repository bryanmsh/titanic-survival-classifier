import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Titanic Survival Classifier — Pipeline & Exploratory Data Analysis (EDA)\n",
                "\n",
                "**Problem:** Tabular Binary Classification (Kaggle Titanic Dataset)  \n",
                "**Target:** `Survived` ($0 = \\text{Died}, 1 = \\text{Survived}$)  \n",
                "**Core Focus:** Strict end-to-end ML workflow, zero data leakage, and rigorous statistical exploration.\n",
                "\n",
                "---\n",
                "\n",
                "## Pipeline Structure\n",
                "1. **Environment Setup & Data Ingestion**\n",
                "2. **Exploratory Data Analysis (EDA)**\n",
                "3. **Data Partitioning & Leakage Prevention**\n",
                "4. **Custom Transformers & Preprocessing Architecture**\n",
                "5. **Feature Engineering & Preprocessing Pipeline Assembly** *(Current Section)*\n",
                "6. **Baseline Benchmark & Multi-Model Cross-Validation**\n",
                "7. **Evaluation Metrics & Asymmetric Error Analysis**\n",
                "8. **Hyperparameter Optimization**\n",
                "9. **Final Test Evaluation & Model Interpretability**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup & Reproducibility\n",
                "We begin by importing our scientific computing libraries and setting a fixed random seed (`RANDOM_STATE = 42`) for deterministic behavior across all stochastic operations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.base import BaseEstimator, TransformerMixin\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
                "from sklearn.impute import SimpleImputer\n",
                "\n",
                "# Set random seed for complete reproducibility\n",
                "RANDOM_STATE = 42\n",
                "np.random.seed(RANDOM_STATE)\n",
                "\n",
                "# Styling parameters for clean, presentation-ready visualizations\n",
                "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
                "plt.rcParams[\"figure.figsize\"] = (9, 5)\n",
                "plt.rcParams[\"axes.titlesize\"] = 13\n",
                "plt.rcParams[\"axes.titleweight\"] = \"bold\"\n",
                "\n",
                "print(\"Environment initialized. Libraries successfully imported.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Ingestion & Integrity Verification\n",
                "\n",
                "We load `data/train.csv` (891 labeled samples) and `data/test.csv` (418 unlabeled samples).\n",
                "\n",
                "> **Zero-Leakage Note:** Following proper statistical methodology, `test.csv` is loaded strictly to verify schema compatibility. It will remain completely untouched throughout feature engineering, imputation, and model selection to prevent **Data Snooping Bias**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_path = os.path.join(\"data\", \"train.csv\")\n",
                "test_path = os.path.join(\"data\", \"test.csv\")\n",
                "\n",
                "train_df = pd.read_csv(train_path)\n",
                "test_df = pd.read_csv(test_path)\n",
                "\n",
                "print(f\"Training dataset shape: {train_df.shape} (891 rows, 12 columns)\")\n",
                "print(f\"Test dataset shape:     {test_df.shape} (418 rows, 11 columns)\")\n",
                "print(f\"Passenger ID uniqueness verified: {train_df['PassengerId'].nunique() == len(train_df)}\")\n",
                "\n",
                "train_df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Schema Diagnosis & Descriptive Statistics\n",
                "\n",
                "Tabular data consists of heterogeneous column types. Let's inspect data types, non-null counts, and statistical summaries:\n",
                "- **Continuous Numerical:** `Age`, `Fare`\n",
                "- **Discrete Numerical:** `SibSp` (# siblings/spouses), `Parch` (# parents/children)\n",
                "- **Ordinal Categorical:** `Pclass` (1st, 2nd, 3rd class)\n",
                "- **Nominal Categorical:** `Sex` ('male', 'female'), `Embarked` ('C', 'Q', 'S')\n",
                "- **Free Text / Identifiers:** `Name`, `Ticket`, `Cabin`\n",
                "- **Binary Target:** `Survived` ($0 = \\text{Perished}, 1 = \\text{Survived}$)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df.info()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df.describe().T"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Statistical Highlights from Summary:\n",
                "1. **`Survived`:** Mean is $0.3838$, meaning 38.38% of passengers survived in the training dataset.\n",
                "2. **`Age`:** Ranges from $0.42$ years (infant) to $80.0$ years. Count is 714, revealing 177 missing values (~20%).\n",
                "3. **`Fare`:** Heavily right-skewed. Median fare is only $\\$14.45$, whereas maximum fare is $\\$512.33$, with standard deviation $\\$49.69$.\n",
                "4. **`SibSp` and `Parch`:** Over 68% traveled with no siblings/spouses, and over 76% traveled with no parents/children."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Target Variable Analysis (`Survived`)\n",
                "\n",
                "Understanding the class balance is critical for defining the **base rate** and choosing appropriate evaluation metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "target_counts = train_df['Survived'].value_counts()\n",
                "target_pct = train_df['Survived'].value_counts(normalize=True) * 100\n",
                "\n",
                "target_summary = pd.DataFrame({\n",
                "    'Count': target_counts,\n",
                "    'Percentage (%)': target_pct.round(2)\n",
                "})\n",
                "target_summary.index = ['Died (0)', 'Survived (1)']\n",
                "print(target_summary)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(6, 4))\n",
                "sns.barplot(x=target_summary.index, y=target_summary['Count'], hue=target_summary.index, palette=['#e74c3c', '#2ecc71'], legend=False, ax=ax)\n",
                "ax.set_title('Target Distribution: Passenger Survival', fontsize=13, fontweight='bold')\n",
                "ax.set_ylabel('Number of Passengers')\n",
                "for p in ax.patches:\n",
                "    h = int(p.get_height())\n",
                "    ax.annotate(f\"{h} ({h/len(train_df)*100:.1f}%)\",\n",
                "                (p.get_x() + p.get_width() / 2., h / 2),\n",
                "                ha='center', va='center', color='white', fontweight='bold', fontsize=12)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Base Rate & The \"Accuracy Paradox\"\n",
                "- **Base Rate:** $P(y=0) = 61.62\\%$, $P(y=1) = 38.38\\%$.\n",
                "- A trivial, non-learning baseline model (`DummyClassifier`) predicting that every single passenger died will achieve **61.62% accuracy**.\n",
                "- **Why Accuracy Alone Fails:** In classification tasks with moderate or severe imbalance, high accuracy can be achieved by completely ignoring the minority class (predicting 0 everywhere yields 0% recall on survivors). Therefore, subsequent steps will emphasize **Precision, Recall, F1-Score, and ROC-AUC**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Demographic Analysis: Sex and Class Gradients\n",
                "\n",
                "Historical maritime evacuation protocol mandated \"women and children first\". Let's verify how strongly `Sex` and `Pclass` determined survival."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))\n",
                "\n",
                "# Survival by Sex\n",
                "sns.barplot(data=train_df, x='Sex', y='Survived', hue='Sex', palette='coolwarm', legend=False, errorbar=None, ax=ax1)\n",
                "ax1.set_title('Survival Rate by Sex', fontsize=12, fontweight='bold')\n",
                "ax1.set_ylabel('Survival Probability')\n",
                "ax1.set_ylim(0, 1.0)\n",
                "for p in ax1.patches:\n",
                "    ax1.annotate(f\"{p.get_height()*100:.1f}%\",\n",
                "                 (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),\n",
                "                 ha='center', va='bottom', fontweight='bold')\n",
                "\n",
                "# Survival by Pclass\n",
                "sns.barplot(data=train_df, x='Pclass', y='Survived', hue='Pclass', palette='Blues_d', legend=False, errorbar=None, ax=ax2)\n",
                "ax2.set_title('Survival Rate by Socioeconomic Class (Pclass)', fontsize=12, fontweight='bold')\n",
                "ax2.set_ylabel('Survival Probability')\n",
                "ax2.set_ylim(0, 1.0)\n",
                "for p in ax2.patches:\n",
                "    ax2.annotate(f\"{p.get_height()*100:.1f}%\",\n",
                "                 (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),\n",
                "                 ha='center', va='bottom', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Interaction Analysis: Sex $\\times$ Pclass\n",
                "Does socioeconomic privilege amplify the gender survival advantage?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "interaction = train_df.groupby(['Pclass', 'Sex'])['Survived'].agg(\n",
                "    Total='count',\n",
                "    Survived='sum',\n",
                "    Survival_Rate=lambda x: f\"{x.mean()*100:.1f}%\"\n",
                ").reset_index()\n",
                "print(\"Interaction Summary:\")\n",
                "print(interaction)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(8, 4.5))\n",
                "sns.barplot(data=train_df, x='Pclass', y='Survived', hue='Sex', \n",
                "            palette={'female': '#e74c3c', 'male': '#3498db'}, errorbar=None, ax=ax)\n",
                "ax.set_title('Survival Rate by Class and Sex Interaction', fontsize=13, fontweight='bold')\n",
                "ax.set_ylabel('Survival Probability')\n",
                "ax.set_ylim(0, 1.15)\n",
                "for p in ax.patches:\n",
                "    h = p.get_height()\n",
                "    if h > 0:\n",
                "        ax.annotate(f\"{h*100:.1f}%\",\n",
                "                    (p.get_x() + p.get_width() / 2., h + 0.02),\n",
                "                    ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
                "plt.legend(title='Sex', loc='upper right')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Empirical Insights:\n",
                "- **Female Priority:** Females in 1st Class (~96.8%) and 2nd Class (~92.1%) had nearly guaranteed survival.\n",
                "- **Steerage Penalty:** In 3rd Class, female survival dropped to 50.0%.\n",
                "- **Male Mortality:** 2nd Class males suffered 84.3% mortality, and 3rd Class males suffered 86.5% mortality. Only 1st Class males had higher survival (~36.9%)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Continuous Feature Analysis: Age and Fare Distributions\n",
                "\n",
                "Let's examine the distributions of `Age` and `Fare` conditioned on survival status."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
                "\n",
                "# Age KDE\n",
                "sns.kdeplot(data=train_df[train_df['Survived'] == 1]['Age'].dropna(), label='Survived (1)', color='#2ecc71', fill=True, ax=ax1)\n",
                "sns.kdeplot(data=train_df[train_df['Survived'] == 0]['Age'].dropna(), label='Died (0)', color='#e74c3c', fill=True, ax=ax1)\n",
                "ax1.set_title('Age Distribution by Survival Outcome', fontsize=12, fontweight='bold')\n",
                "ax1.set_xlabel('Age (Years)')\n",
                "ax1.legend()\n",
                "\n",
                "# Fare Distribution (Log Scale to handle positive skew)\n",
                "sns.boxplot(data=train_df, x='Survived', y='Fare', hue='Survived', palette=['#e74c3c', '#2ecc71'], legend=False, ax=ax2)\n",
                "ax2.set_yscale('log')\n",
                "ax2.set_title('Log-Scaled Fare Distribution by Survival', fontsize=12, fontweight='bold')\n",
                "ax2.set_xticks([0, 1])\n",
                "ax2.set_xticklabels(['Died (0)', 'Survived (1)'])\n",
                "ax2.set_ylabel('Fare (Log Scale)')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Family Dynamics & Group Survival (`SibSp`, `Parch`)\n",
                "\n",
                "Did family traveling companions affect survival odds?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Exploring family size dynamics\n",
                "temp_df = train_df.copy()\n",
                "temp_df['FamilySize'] = temp_df['SibSp'] + temp_df['Parch'] + 1\n",
                "temp_df['IsAlone'] = (temp_df['FamilySize'] == 1).astype(int)\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))\n",
                "\n",
                "sns.barplot(data=temp_df, x='FamilySize', y='Survived', hue='FamilySize', palette='magma', legend=False, errorbar=None, ax=ax1)\n",
                "ax1.set_title('Survival Rate vs. Total Family Size', fontsize=12, fontweight='bold')\n",
                "ax1.set_ylabel('Survival Probability')\n",
                "ax1.set_ylim(0, 0.8)\n",
                "\n",
                "sns.barplot(data=temp_df, x='IsAlone', y='Survived', hue='IsAlone', palette='Set2', legend=False, errorbar=None, ax=ax2)\n",
                "ax2.set_title('Survival Rate: Solo Traveler vs. Accompanied', fontsize=12, fontweight='bold')\n",
                "ax2.set_xticks([0, 1])\n",
                "ax2.set_xticklabels(['With Family (0)', 'Solo Traveler (1)'])\n",
                "ax2.set_ylabel('Survival Probability')\n",
                "ax2.set_ylim(0, 0.8)\n",
                "for p in ax2.patches:\n",
                "    ax2.annotate(f\"{p.get_height()*100:.1f}%\",\n",
                "                 (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),\n",
                "                 ha='center', va='bottom', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Finding:\n",
                "- Solo travelers had lower survival (~30.4%) compared to passengers traveling with moderate families (~50.6%).\n",
                "- However, very large families (5+) experienced severe survival collapse ($< 20\\%$), due to logistical panic keeping large parties intact.\n",
                "- This justifies creating `FamilySize` and `IsAlone` features during feature engineering."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Missing Data Audit & Strategic Treatment Plan\n",
                "\n",
                "Handling missing data properly is essential to prevent bias and data leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "missing_counts = train_df.isnull().sum()\n",
                "missing_pct = (missing_counts / len(train_df)) * 100\n",
                "missing_table = pd.DataFrame({\n",
                "    'Missing Values': missing_counts,\n",
                "    'Missing Percentage (%)': missing_pct.round(2)\n",
                "})\n",
                "missing_table = missing_table[missing_table['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)\n",
                "print(\"Missing Value Audit:\")\n",
                "print(missing_table)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(7, 3.5))\n",
                "sns.barplot(x=missing_table.index, y=missing_table['Missing Percentage (%)'], hue=missing_table.index, palette='Reds_r', legend=False, ax=ax)\n",
                "ax.set_title('Missing Value Percentage by Feature', fontsize=12, fontweight='bold')\n",
                "ax.set_ylabel('Missing (%)')\n",
                "ax.set_ylim(0, 100)\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f\"{p.get_height():.1f}%\",\n",
                "                 (p.get_x() + p.get_width() / 2., p.get_height() + 2),\n",
                "                 ha='center', va='bottom', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Comprehensive Strategy Table for Missing Data\n",
                "\n",
                "| Feature | Missing (%) | Missingness Type | Planned Strategy | Technical & Domain Justification |\n",
                "|---|---|---|---|---|\n",
                "| **`Age`** | 19.87% (177 rows) | MAR (Missing at Random) | **Conditional Imputation** (Median by `Title` & `Pclass`) | Dropping 177 rows discards 20% of data. Global median (28.0) discards known age hierarchy (Master = ~4 yrs, Mr = ~30 yrs). Conditioned median preserves subpopulation variance without leakage when computed on train split. |\n",
                "| **`Cabin`** | 77.10% (687 rows) | MNAR (Missing Not at Random) | **Binary Indicator** (`HasCabin`) | 77% missing. Rather than dropping or fabricating cabin letters, missingness itself is predictive: 1st class passengers had cabins (78% recorded); 3rd class steerage had unrecorded berths (only 3% recorded). |\n",
                "| **`Embarked`** | 0.22% (2 rows) | MCAR (Missing Completely at Random) | **Mode Imputation** (`'S'`) | Only 2 rows missing. Imputing with the statistical mode introduces zero noticeable distortion. |\n",
                "| **`Fare`** | 0% in train, 1 in test | MAR | **Median by `Pclass`** | Handled inside the pipeline so unseen test data with missing fare processes seamlessly. |"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Feature Engineering Feasibility: Honorific Title Extraction\n",
                "\n",
                "Let's test the extraction of passenger titles from `Name` to assess its predictive utility."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "extracted_titles = train_df['Name'].str.extract(r',\\s*([^\\.]+)\\.', expand=False)\n",
                "train_df['Title'] = extracted_titles\n",
                "\n",
                "# Group rare titles\n",
                "common_titles = ['Mr', 'Mrs', 'Miss', 'Master']\n",
                "train_df['TitleGroup'] = train_df['Title'].apply(lambda x: x if x in common_titles else 'Rare')\n",
                "\n",
                "title_summary = train_df.groupby('TitleGroup')['Survived'].agg(\n",
                "    Count='count',\n",
                "    Survival_Rate=lambda x: f\"{x.mean()*100:.1f}%\"\n",
                ").sort_values(by='Count', ascending=False)\n",
                "\n",
                "print(\"Title Group Distribution & Survival Rate:\")\n",
                "print(title_summary)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(7, 4))\n",
                "sns.barplot(data=train_df, x='TitleGroup', y='Survived', hue='TitleGroup', palette='viridis', legend=False, errorbar=None, ax=ax)\n",
                "ax.set_title('Survival Rate by Honorific Title Group', fontsize=12, fontweight='bold')\n",
                "ax.set_ylabel('Survival Probability')\n",
                "ax.set_ylim(0, 1.0)\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f\"{p.get_height()*100:.1f}%\",\n",
                "                 (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),\n",
                "                 ha='center', va='bottom', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Finding:\n",
                "`TitleGroup` cleanly isolates:\n",
                "- **`Mrs` & `Miss`:** ~79% and ~69% survival.\n",
                "- **`Master` (young boys):** ~57% survival (significantly higher than adult `Mr` at ~15%).\n",
                "- **`Mr`:** ~15% survival.\n",
                "- **`Rare` (nobility/officers/clergy):** ~34% survival.\n",
                "\n",
                "This confirms that Title extraction provides strong predictive signal."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Data Partitioning & Zero-Leakage Protocol\n",
                "\n",
                "### The Imperative of Zero Data Leakage\n",
                "**Data leakage** occurs when information from outside the training dataset (such as the validation fold, test set, or target variable) is used during feature preprocessing, parameter estimation, or model selection. \n",
                "\n",
                "A subtle but widespread blunder in tabular ML is computing global preprocessing parameters (e.g. median age, category frequencies, scaling means) across the complete dataset *before* splitting. Doing so allows information about the test distribution to contaminate model training, creating artificially optimistic cross-validation results that fail in production.\n",
                "\n",
                "### Stratified Splitting Strategy\n",
                "Because our target variable `Survived` is moderately imbalanced (61.6% deceased vs. 38.4% survived), a purely random split risks sampling variance where target proportions drift between folds. We employ **Stratified Splitting** to enforce identical target distributions across partitions:\n",
                "- **Training Partition (`X_train`, `y_train`):** 80% (712 passengers) — used for all transformer fitting, cross-validation, and hyperparameter tuning.\n",
                "- **Held-Out Test Partition (`X_test`, `y_test`):** 20% (179 passengers) — strictly isolated and untouched until final evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Isolate feature matrix X and target vector y\n",
                "X = train_df.drop(columns=['Survived', 'PassengerId'])\n",
                "y = train_df['Survived']\n",
                "\n",
                "# Stratified 80/20 train/test split\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y\n",
                ")\n",
                "\n",
                "print(f\"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\")\n",
                "print(f\"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Verify stratification integrity across splits\n",
                "strat_check = pd.DataFrame({\n",
                "    'Original Population (%)': y.value_counts(normalize=True) * 100,\n",
                "    'Training Split (%)': y_train.value_counts(normalize=True) * 100,\n",
                "    'Held-Out Test (%)': y_test.value_counts(normalize=True) * 100\n",
                "})\n",
                "strat_check.index = ['Died (0)', 'Survived (1)']\n",
                "print(\"Stratification Verification:\")\n",
                "print(strat_check.round(2))\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(7, 4))\n",
                "strat_check.plot(kind='bar', ax=ax, colormap='Set2', width=0.7)\n",
                "ax.set_title('Target Distribution Across Partitions (Stratification Check)', fontsize=13, fontweight='bold')\n",
                "ax.set_ylabel('Percentage of Samples (%)')\n",
                "ax.set_ylim(0, 80)\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f\"{p.get_height():.1f}%\",\n",
                "                (p.get_x() + p.get_width() / 2., p.get_height() + 1.2),\n",
                "                ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
                "plt.xticks(rotation=0)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Custom Scikit-Learn Preprocessing Transformers\n",
                "\n",
                "To guarantee zero data leakage and enable seamless pipelining, we implement custom transformers that strictly adhere to Scikit-Learn's **Estimator & Transformer Contract**:\n",
                "\n",
                "1. **`fit(X, y=None)`:** Learns parameter states (such as group medians or common category lists) exclusively from `X_train` and stores them as internal attributes.\n",
                "2. **`transform(X)`:** Applies deterministic transformations using the stored parameters to any incoming dataset without recalculating or modifying state.\n",
                "\n",
                "### Transformers Implemented:\n",
                "- **`TitleExtractor`:** Extracts titles from `Name` and groups infrequent titles into `'Rare'` based on training frequency.\n",
                "- **`FamilyFeaturesAdder`:** Creates `FamilySize = SibSp + Parch + 1` and `IsAlone = (FamilySize == 1)`.\n",
                "- **`CabinIndicator`:** Extracts `HasCabin` indicator.\n",
                "- **`GroupedAgeImputer`:** Imputes missing `Age` values using conditional medians calculated within each `(TitleGroup, Pclass)` cohort learned on `X_train`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.transformers import (\n",
                "    TitleExtractor,\n",
                "    FamilyFeaturesAdder,\n",
                "    CabinIndicator,\n",
                "    GroupedAgeImputer\n",
                ")\n",
                "\n",
                "# Instantiate transformers\n",
                "title_extractor = TitleExtractor()\n",
                "family_adder = FamilyFeaturesAdder()\n",
                "cabin_indicator = CabinIndicator()\n",
                "age_imputer = GroupedAgeImputer()\n",
                "\n",
                "# Fit transformers strictly on X_train\n",
                "age_imputer.fit(X_train)\n",
                "print(\"Grouped medians learned strictly from X_train:\")\n",
                "for k, v in sorted(age_imputer.group_medians_.items()):\n",
                "    print(f\"  Title: {k[0]:<7} | Pclass: {k[1]} -> Median Age: {v:.1f}\")\n",
                "print(f\"Fallback Global Median (from X_train): {age_imputer.global_median_:.1f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Unified ColumnTransformer & Full Preprocessing Pipeline\n",
                "\n",
                "Now we combine our custom transformers and standard Scikit-Learn transformers into a single composite pipeline. We define three parallel preprocessing branches via **`ColumnTransformer`**:\n",
                "\n",
                "1. **Numeric Branch (`['Age', 'Fare', 'FamilySize', 'Pclass']`):**\n",
                "   - `SimpleImputer(strategy='median')`: Handles any remaining missing values (such as the single missing `Fare` row in `test.csv`).\n",
                "   - `StandardScaler()`: Standardizes features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). Essential for gradient descent convergence and coefficient interpretation in regularized linear models (Logistic Regression).\n",
                "2. **Categorical Branch (`['Sex', 'Embarked', 'TitleGroup']`):**\n",
                "   - `SimpleImputer(strategy='most_frequent')`: Imputes missing ports with mode `'S'`.\n",
                "   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Converts nominal levels into binary dummy columns. Any unseen categories encountered in future data are gracefully handled without crashing.\n",
                "3. **Binary Branch (`['IsAlone', 'HasCabin']`):**\n",
                "   - Passed through directly as clean 0/1 binary indicators.\n",
                "4. **Drop Uninformative Features:**\n",
                "   - High-cardinality text columns (`Name`, `Ticket`, `Cabin`) and raw components (`SibSp`, `Parch`) are dropped from the final design matrix."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.pipeline import build_preprocessor\n",
                "\n",
                "# Instantiate the end-to-end preprocessing pipeline\n",
                "preprocessor = build_preprocessor(scale_numeric=True)\n",
                "print(\"Preprocessor Pipeline Architecture:\")\n",
                "print(preprocessor)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Pipeline Fitting & Design Matrix Inspection\n",
                "\n",
                "We fit the entire composite preprocessor **strictly on `X_train`** and transform both `X_train` and `X_test`.\n",
                "\n",
                "Let's inspect the resulting numerical design matrix, column names, and check for any remaining missing values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit and transform training features strictly on X_train\n",
                "X_train_trans = preprocessor.fit_transform(X_train)\n",
                "\n",
                "# Transform unseen test partition without refitting\n",
                "X_test_trans = preprocessor.transform(X_test)\n",
                "\n",
                "# Retrieve generated feature names from ColumnTransformer\n",
                "col_transformer = preprocessor.named_steps['column_transform']\n",
                "feature_names = col_transformer.get_feature_names_out()\n",
                "\n",
                "print(f\"Processed X_train matrix shape: {X_train_trans.shape} (712 samples, {len(feature_names)} features)\")\n",
                "print(f\"Processed X_test matrix shape:  {X_test_trans.shape} (179 samples, {len(feature_names)} features)\")\n",
                "print(f\"Total NaN count in X_train: {np.isnan(X_train_trans).sum()}\")\n",
                "print(f\"Total NaN count in X_test:  {np.isnan(X_test_trans).sum()}\")\n",
                "\n",
                "# Construct DataFrame for inspection\n",
                "X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)\n",
                "print(\"\\nTransformed Design Matrix (First 5 Rows):\")\n",
                "display(X_train_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Distribution of Standardized Numeric Features\n",
                "Let's verify that `StandardScaler` successfully centered the continuous variables (`Age`, `Fare`, `FamilySize`) around $\\mu = 0$ with $\\sigma = 1$ on the training split."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
                "\n",
                "sns.kdeplot(X_train_df['num__Age'], fill=True, color='#3498db', ax=axes[0])\n",
                "axes[0].set_title(f\"Standardized Age\\n(Mean: {X_train_df['num__Age'].mean():.2f}, Std: {X_train_df['num__Age'].std():.2f})\", fontsize=11, fontweight='bold')\n",
                "\n",
                "sns.kdeplot(X_train_df['num__Fare'], fill=True, color='#2ecc71', ax=axes[1])\n",
                "axes[1].set_title(f\"Standardized Fare\\n(Mean: {X_train_df['num__Fare'].mean():.2f}, Std: {X_train_df['num__Fare'].std():.2f})\", fontsize=11, fontweight='bold')\n",
                "\n",
                "sns.kdeplot(X_train_df['num__FamilySize'], fill=True, color='#9b59b6', ax=axes[2])\n",
                "axes[2].set_title(f\"Standardized Family Size\\n(Mean: {X_train_df['num__FamilySize'].mean():.2f}, Std: {X_train_df['num__FamilySize'].std():.2f})\", fontsize=11, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. Summary & Modeling Readiness\n",
                "\n",
                "### Feature Pipeline Accomplishments:\n",
                "1. **Complete Preprocessing Encapsulation:** Unified custom feature extraction (`TitleGroup`, `FamilySize`, `IsAlone`, `HasCabin`, grouped `Age` imputation) and standard transformations into a single, leak-free `Pipeline`.\n",
                "2. **Zero Missingness:** Guaranteed that both train and held-out test splits contain zero missing values across all 16 design matrix columns.\n",
                "3. **Standardized Numerical Scale:** Continuous and discrete numeric features are standardized to $\\mathcal{N}(0, 1)$ without distorting categorical indicator columns.\n",
                "4. **One-Hot Encoding with Unknown Handling:** Categoricals (`Sex`, `Embarked`, `TitleGroup`) are cleanly encoded with `handle_unknown='ignore'`, preventing test-time dimension mismatch.\n",
                "\n",
                "### Next Steps:\n",
                "- Construct the non-learning benchmark baseline (`DummyClassifier`).\n",
                "- Train and evaluate three distinct model families using Stratified 5-Fold Cross-Validation: **Logistic Regression**, **Random Forest Classifier**, and **Gradient Boosting Classifier**.\n",
                "- Compare models across multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score, and ROC-AUC)."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("titanic_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("titanic_pipeline.ipynb successfully generated with unified preprocessing pipeline!")
