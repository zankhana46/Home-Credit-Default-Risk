# Home Credit Default Risk Prediction

This project builds an end-to-end machine learning pipeline to predict the probability of a loan applicant defaulting on their repayment, using data from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk).

---

## 📌 Objective

Traditional credit scoring methods often fail to evaluate applicants with limited credit history. This project aims to develop a robust, interpretable ML pipeline that:
- Predicts default risk using structured application and credit history data
- Handles class imbalance and missing values
- Provides actionable insights for financial decision-making

---

## 📂 Repository Contents
```
├── data/ #contains all the tables from dataset and new generated ones from processing files
├── EDA.ipynb # Exploratory Data Analysis
├── Feature_engineering.ipynb # joining important aggregated features from different datasets
├── Data_preprocessing.ipynb # Cleaning, Imputation, Encoding
├── Baseline_model.ipynb # Initial model using apllication train data
├── Model_Training_Eval.ipynb # Final models with evaluation and tuning
├── Home credit default .pptx # Presenatation summarizing my methodology and results
├── Table representations.png # Image showing the different datasets and their keys
└── README.md # Project overview and documentation
```

## 🧠 Methodology

### 1. **EDA**
- Identified skewed distributions and key missing features
- Found target imbalance (~8% defaulters)
- Insights: Defaulters tend to be younger, low-income, in unstable jobs

### 2. **Feature Engineering**
- Created financial ratios (credit-to-income, annuity-to-income, etc.)
- Aggregated external sources like `bureau`, `POS_CASH`, `installments`
- Final dataset contains 167 relevant features

### 3. **Preprocessing**
- Removed features with >60% missing
- Imputed numeric (median) and categorical (mode) values
- Applied consistent Label Encoding across train/valid/test sets
- - Removed highly correlated and low-importance features

### 4. **Modeling**
- Compared: `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`
- Metric: ROC-AUC (suitable for imbalanced classification)
- CV: StratifiedKFold for robust evaluation

### 5. **Optimization**
- Used:
  - Top-K feature selection (top 50)
  - Stratified K-Fold CV
  - GridSearchCV for hyperparameter tuning

---

## 📊 Results

### 📈 ROC-AUC Comparison:

| Model           | Baseline (Raw) | Final (Preprocessed) |
|----------------|----------------|-----------------------|
| Random Forest   | 0.7119         | 0.7344                |
| XGBoost         | 0.7646         | 0.7673                |
| LightGBM        | 0.7641         | 0.7723                |
| **CatBoost**    | 0.7589         | **0.7768**            |

### 🔧 Optimization (CatBoost):

| Approach                  | ROC-AUC     |
|--------------------------|-------------|
| Stratified K-Fold CV     | 0.7767      |
| Top 50 features only     | 0.7696      |
| GridSearchCV (tuned)     | **0.7778**  |

---

## 📁 Output

- Final submission CSV contains:
  - `SK_ID_CURR`  
  - `TARGET` (predicted probability of default)
- File created using: `CatBoost.predict_proba(X_test)[:, 1]`
- File location /data/submission_catboost.csv

---

## 🔁 How to Reproduce the Results

1. Clone the repo
2. Run notebooks in order:
   - `EDA.ipynb`
   - `Baseline_model.ipynb`
   - `Feature_engineering.ipynb`
   - `Data_preprocessing.ipynb`
   - `Model_Training_Eval.ipynb`
3. (Optional) Submit predictions from `final_submission.csv` on [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/submit)

---

## 📚 Acknowledgements

- [Home Credit](https://www.homecredit.net/)
- [Kaggle Dataset](https://www.kaggle.com/competitions/home-credit-default-risk)
- Inspired by solutions from the Kaggle community





