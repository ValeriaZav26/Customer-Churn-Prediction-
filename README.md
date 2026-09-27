# Customer Churn Prediction

Binary classification project predicting customer churn on the **Telco Customer Churn** dataset, combining exploratory data analysis with two tree-based models (Random Forest and XGBoost) tuned via grid search.

## Overview

Customer churn — the loss of subscribers — is one of the most direct levers on revenue for subscription-based businesses (telecom, SaaS, streaming). This project analyzes churn drivers through EDA and builds a predictive model that estimates the probability that a given customer will leave, so retention efforts can be targeted at high-risk segments.

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, also widely mirrored on Kaggle)
- **Rows:** ~7,000 customers
- **Target:** `Churn` (Yes/No) — binary, moderately imbalanced (~26% churn rate)
- **Features:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account info (`tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), and subscribed services (`InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.)

## Exploratory Data Analysis

Key findings from the EDA section:

- **Tenure** has the strongest correlation with churn — risk is highest in the first few months and drops sharply as tenure increases.
- **Contract type** is a major driver: month-to-month customers churn far more than one-year or two-year contract holders.
- **Payment method** matters — customers paying via electronic check churn noticeably more than those on automatic bank transfer or credit card.
- **Add-on services** (online security, tech support) correlate with lower churn, suggesting engaged/invested customers are more likely to stay.
- **Monthly charges** correlate positively with churn, while accumulated `TotalCharges` correlates negatively (a proxy for tenure/loyalty).

Visualizations include boxplots for charge distributions, a churn-rate breakdown by payment method and contract type, a KDE plot of tenure vs. churn, and a full correlation heatmap.

## Modeling

- **Preprocessing:** categorical encoding and missing-value imputation are handled inside an `sklearn` `Pipeline` / `ColumnTransformer`, fit only on the training split — no information from the validation set leaks into preprocessing or feature encoding.
- **Models:**
  - `RandomForestClassifier`
  - `XGBClassifier`
- **Hyperparameter tuning:** `GridSearchCV` (5-fold, stratified) optimizing ROC-AUC, over depth, number of estimators, regularization, sampling ratios, and class-imbalance weighting (`class_weight` / `scale_pos_weight`).
- **Evaluation metrics:** ROC-AUC, PR-AUC, and full classification report (precision/recall/F1), chosen to reflect the class imbalance rather than relying on accuracy alone.

### Results

| Model | Valid ROC-AUC | Notes |
|---|---|---|
| Random Forest | ~0.83 | Larger train/valid gap (more overfitting) |
| XGBoost | ~0.83 | Smaller train/valid gap — selected as the final model |

Recall on the churn class is moderate (~0.48) at the default 0.5 threshold, meaning the model currently misses a substantial share of churners. For a retention use case, the classification threshold should be tuned against the business cost of a missed churner vs. an unnecessary retention offer, rather than left at 0.5.

## Project Structure

```
.
├── Customer_Churn_Prediction_.ipynb   # main notebook: EDA + modeling
├── README.md
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # not included — see Setup
```

## Setup & Usage

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV in the project directory (update the file path in the first cell if needed).
3. Run the notebook top to bottom in Jupyter, JupyterLab, or Google Colab.

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost`

