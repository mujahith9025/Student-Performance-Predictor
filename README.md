# 🎓 Student Performance Predictor (Advanced Machine Learning)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project designed to predict student examination marks (Mathematics) based on demographic factors, socio-economic indicators, study habits, and existing subject competencies using **Hyperparameter-Tuned Gradient Boosting and Meta-Ensembles**.

---

## 📌 1. Project Overview & Problem Statement
Understanding the factors that influence academic success allows educators and parents to identify at-risk students early and provide targeted academic interventions. 

This project implements an advanced **Supervised Machine Learning Regression Pipeline**:
* **Input Features:** Gender, Ethnicity, Parental Level of Education, Lunch Type, Test Preparation Course, Reading Score, Writing Score.
* **Target Variable:** Mathematics Examination Marks ($0 - 100$).
* **Cross-Validation:** 5-Fold GridSearch Cross-Validation.
* **Champion Model:** **Voting Ensemble Regressor** achieving **$76.44\%$ $R^2$ accuracy** and an average prediction error of **$\pm 5.95$ marks**.

---

## 🏗️ 2. Project Architecture & Workflow

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Phase 1: Dataset Setup │ ──► │     Phase 2: EDA        │ ──► │  Phase 3: Preprocessing │
│ (1,000 Student Records) │     │ (Distributions & Corrs) │     │ (ColumnTransformer OHE) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐                  │
│    Phase 6: Web App     │ ◄── │   Phase 5: Evaluation   │ ◄────────────────┘
│ (Streamlit Interactive) │     │ (14 Model Leaderboard)  │     Phase 4: Hyperparameter Tuning
└─────────────────────────┘     └─────────────────────────┘     (5-Fold CV GridSearch on Boosting & Ensembles)
```

---

## 📊 3. Exploratory Data Analysis (EDA) Insights

From analyzing the 1,000 student dataset:
1. **Test Preparation Course (+9.4 marks):**
   * Students who completed the pre-exam preparation course scored **$74.24$ average in Math** vs. **$64.84$** for those who did not.
2. **Nutritional Support (+8.0 marks):**
   * Students receiving standard lunch scored **$70.97$** compared to **$62.93$** for students on free/reduced lunch.
3. **Parental Education Impact:**
   * Students whose parents hold a Master's degree achieved the highest scores (**$75.32$ average**).
4. **Subject Synergy:**
   * Reading and Writing scores exhibit high correlation ($r \approx 0.95$), and both correlate strongly with Math score ($r \approx 0.80$).

---

## 🏆 4. Model Comparison & Hyperparameter Tuning Results

We evaluated 14 baseline, tuned, and ensemble algorithms on 200 unseen test students:

| Rank | Model Name | Model Type | Test $R^2$ Accuracy | Test MAE *(Avg Error)* | Test RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **Voting Ensemble Regressor** | Meta-Ensemble (Ridge + GB + RF) | **76.44%** | **$\pm$ 5.95 marks** | **7.56** |
| 🥈 | **Tuned Ridge Regression** | Regularized Linear ($\alpha=10.0$) | **76.31%** | **$\pm$ 5.98 marks** | **7.58** |
| 🥉 | **Tuned ElasticNet** | L1+L2 Regularization | **76.14%** | $\pm$ 5.99 marks | 7.61 |
| 4 | **Tuned Lasso Regression** | L1 Regularization | **76.04%** | $\pm$ 5.99 marks | 7.62 |
| 5 | **Stacking Ensemble Regressor** | Meta-Learner (Ridge Stacking) | **76.03%** | $\pm$ 6.01 marks | 7.62 |
| 6 | **Tuned Gradient Boosting** | Boosting Trees ($lr=0.08$) | **75.53%** | $\pm$ 6.07 marks | 7.70 |
| 7 | **Tuned Random Forest** | Bagging Ensemble ($d=6$) | **75.03%** | $\pm$ 6.14 marks | 7.78 |
| 8 | **Tuned AdaBoost** | Adaptive Boosting | **74.41%** | $\pm$ 6.23 marks | 7.88 |

---

## 📂 5. Directory Structure

```
student-performance-predictor/
│
├── artifacts/                     # Serialized preprocessors & ML models
│   ├── best_model.joblib          # Champion trained model (Voting Ensemble)
│   ├── preprocessor.joblib        # Scikit-Learn ColumnTransformer pipeline
│   ├── model_metrics.csv          # Comprehensive evaluation metrics across 14 models
│   ├── hyperparameter_tuning_results.csv # 5-Fold CV Tuning parameters
│   └── models/                    # Checkpoints for all trained algorithms
│
├── data/
│   ├── StudentsPerformance.csv    # 1,000 student raw dataset
│   └── processed/                 # ML-ready train/test numpy matrices
│
├── plots/                         # Generated EDA and evaluation charts
│   ├── 01_score_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_test_prep_impact.png
│   ├── 04_parental_education_impact.png
│   ├── 05_lunch_impact.png
│   ├── 06_model_performance_comparison.png
│   └── 07_actual_vs_predicted.png
│
├── src/                           # Pipeline source code
│   ├── generate_data.py           # Dataset generator
│   ├── eda.py                     # Exploratory Data Analysis & visualizer
│   ├── preprocessing.py           # Feature engineering & train-test split
│   ├── train_models.py            # Baseline model training
│   ├── hyperparameter_tuning.py   # 5-Fold CV GridSearchCV & Ensemble builder
│   ├── evaluate_models.py         # Testing, metric computation & leaderboard
│   └── test_inference.py          # End-to-end inference verification
│
├── app.py                         # Interactive Multi-Tab Streamlit Application
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 🚀 6. Installation & How to Run

### Step 1: Clone or Navigate to Project Directory
```bash
git clone https://github.com/mujahith9025/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Complete Machine Learning Pipeline
```bash
# 1. Run EDA & generate visual charts
python src/eda.py

# 2. Run Data Preprocessing & ColumnTransformer
python src/preprocessing.py

# 3. Run Hyperparameter Tuning with 5-Fold Cross Validation
python src/hyperparameter_tuning.py

# 4. Evaluate models and build leaderboard
python src/evaluate_models.py
```

### Step 4: Launch Interactive Web App
```bash
python -m streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows).*

---
*Created as part of the Advanced Machine Learning Portfolio.*
