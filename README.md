# 📉 Customer Churn Prediction — Telco Dataset

Predicting which telecom customers are likely to churn using EDA and ML classification models, so retention teams can intervene before customers leave.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006400?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

---

## 🗂️ Structure

```
├── Churn_predict.ipynb                   # EDA, feature engineering, modeling
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── README.md
└── requirements.txt
```

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 rows × 21 columns. Target: `Churn` (Yes/No), imbalanced **73.5% retained / 26.5% churned**. Features cover demographics, account info (tenure, contract, charges), and subscribed services. No missing values or duplicates.

## 🔍 Key EDA Insights

- **Contract type** is the strongest driver: month-to-month churns at **42.7%** vs. **11.3%** (1-year) and **2.8%** (2-year).
- **Electronic Check** payers churn at **45.3%**, vs. ~15–19% for other payment methods.
- Churn risk peaks around **month 3** of tenure, then steadily declines.
- Strongest correlations with churn: `tenure` (**-0.35**), `MonthlyCharges` (**+0.19**), `TotalCharges` (**-0.20**). Security/support add-ons and having a partner/dependents reduce churn risk.

## 🛠️ Preprocessing & Feature Engineering

- Dropped `customerID`; binary-encoded Yes/No columns; collapsed "No internet/phone service" categories to `0`.
- Fixed `TotalCharges` dtype (`object` → `float`, missing filled with `0`).
- One-hot encoded categorical features via `ColumnTransformer`.
- **Leakage control:** train/valid split before fitting; preprocessing + model combined in one `sklearn.Pipeline` so encoders are fit only on training folds.

## 🤖 Models & Results

Tuned with `RandomizedSearchCV` (5-fold CV) inside the pipeline:

| Model | Valid ROC-AUC | Valid PR-AUC | Accuracy | Churn F1 |
|---|---|---|---|---|
| **Random Forest** 🏆 | **0.832** | **0.636** | **0.792** | **0.557** |
| XGBoost | 0.810 | 0.602 | 0.779 | 0.540 |

Accuracy alone is misleading given class imbalance, so ROC-AUC/PR-AUC and per-class F1 were used to pick the model. **Random Forest** wins on all key metrics. Both models show train/valid gaps (RF: 0.927→0.832) suggesting some overfitting to address next.

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook Churn_predict.ipynb
```

**`requirements.txt`**: `pandas numpy matplotlib seaborn scikit-learn xgboost scipy jupyter`
