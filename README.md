# 🎓 Student Performance & Dropout Risk Predictor (Dual-Engine ML + XAI + Goal Simulator)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Reports-ReportLab_5.0-red?logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)
[![XAI](https://img.shields.io/badge/Explainable_AI-SHAP_Attributions-purple)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning system designed to predict student examination marks, classify **Academic Pass / Fail & Dropout Risk**, generate **Verified PDF Report Cards**, explain every prediction using **Explainable AI (XAI)**, and reverse-engineer academic target pathways using a **"What-If" Academic Goal Simulator**.

---

## 📌 1. Project Overview & Capabilities
Understanding academic trajectory early allows educational institutions to deploy targeted interventions before exams take place.

This project implements a multi-capability Machine Learning pipeline:
1. **Task A (Regression):** Predicts exact mathematical marks ($0 - 100$) using a **Voting Meta-Ensemble** (**$76.44\%$ $R^2$ accuracy**, $\pm 5.95$ marks average error).
2. **Task B (Classification):** Forecasts **Pass / Fail & Dropout Risk Probability** using **Support Vector Machines** (**$89.5\%$ accuracy, $0.933$ ROC-AUC score**).
3. **Task C ('What-If' Goal Simulator):** Reverse-engineers the minimum required reading and writing milestone scores and study interventions to achieve any target grade.
4. **Task D (Explainable AI):** Provides **Global Permutation Feature Importance** and **Local Real-Time SHAP Attributions** explaining exact points added or deducted.
5. **Task E (Verified Reporting):** Generates and downloads verified, styled **PDF Academic Performance Reports** in one click.

---

## 🎯 2. 'What-If' Academic Goal Simulator

The **Goal Simulator** solves the reverse optimization problem:
* **Student Target Input:** Desired Math Score (e.g. $85$ marks) or Target Grade (Grade A/A+).
* **Current Baseline:** Current reading ($65$) and writing ($62$) scores.
* **Simulator Output Roadmap:**
  * **Gap to Bridge:** $+22.9$ marks.
  * **Step 1:** Complete Test Preparation Course (delivers $+9.4$ marks statistical boost).
  * **Step 2:** Milestone Target: Reading Score $\rightarrow 90 / 100$ ($+25$ marks) & Writing Score $\rightarrow 87 / 100$ ($+25$ marks).
  * **Feasibility Rating:** Classified as *Easily Achievable*, *Moderate Effort*, or *Intensive Intervention*.

---

## 🔍 3. Explainable AI (XAI) & Global Importance Rankings

| Rank | Feature | Relative Impact | Key Finding |
| :---: | :--- | :---: | :--- |
| 🥇 | **Writing Score** | **41.03%** | Writing proficiency strongly reinforces analytical and mathematical problem solving. |
| 🥈 | **Reading Score** | **36.25%** | Reading comprehension is essential for understanding complex examination questions. |
| 🥉 | **Gender Baseline** | **21.46%** | Historical variance baseline across subject domains. |
| 4 | **Test Prep Course** | **0.65%** | Completing the course adds **$+9.4$ marks** average boost. |
| 5 | **Lunch Nutrition Plan** | **0.03%** | Standard nutritional support adds **$+8.0$ marks** average boost. |

---

## 🏗️ 4. Multi-Engine Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│               1. Raw Student Profile Data              │
│    (Demographics + Environment + Reading/Writing Marks)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     2. ColumnTransformer Preprocessing Pipeline        │
│   (StandardScaler for Scores + OneHotEncoder for Cats) │
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ A. Regressor    │ │ B. Classifier   │ │ C. Simulator    │
│ Voting Ensemble │ │ Support Vector  │ │ 'What-If' Target│
│ Peak R²: 76.44% │ │ ROC-AUC: 0.9326 │ │ Milestone Solver│
│ MAE: ±5.95 marks│ │ Accuracy: 89.5% │ │ Gap Roadmapping │
└────────────┬────┘ └────────┬────────┘ └────────┬────────┘
             │               │                   │
             └───────────────┼───────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│   4. Streamlit Dashboard & Verified PDF Generator      │
│  (Live Prediction + Risk Tiers + Roadmap + PDF Export) │
└────────────────────────────────────────────────────────┘
```

---

## 🏆 5. Model Leaderboards

### **A. Regression Leaderboard (Continuous Marks Forecast)**

| Rank | Model Name | Architecture | Test $R^2$ | Test MAE | Test RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **Voting Ensemble Regressor** | Meta-Ensemble (Ridge + GB + RF) | **76.44%** | **$\pm$ 5.95** | **7.56** |
| 🥈 | **Tuned Ridge Regression** | Regularized Linear ($\alpha=10.0$) | **76.31%** | $\pm$ 5.98 | 7.58 |
| 🥉 | **Tuned ElasticNet** | L1+L2 Regularization | **76.14%** | $\pm$ 5.99 | 7.61 |
| 4 | **Tuned Lasso Regression** | L1 Shrinkage | **76.04%** | $\pm$ 5.99 | 7.62 |
| 5 | **Stacking Ensemble Regressor** | Meta-Learner (Ridge Stacking) | **76.03%** | $\pm$ 6.01 | 7.62 |

---

### **B. Classification Leaderboard (Pass / Fail & Dropout Risk)**

| Rank | Classifier Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Support Vector Classifier (SVC)** | **89.50%** | **0.9205** | **0.9586** | **0.9391** | **0.9326** |
| 🥈 | **Logistic Regression** | 89.00% | 0.9249 | 0.9467 | 0.9357 | 0.9320 |
| 🥉 | **Gradient Boosting Classifier** | 90.50% | 0.9310 | 0.9586 | 0.9446 | 0.9272 |
| 4 | **Random Forest Classifier** | 90.00% | 0.9209 | 0.9645 | 0.9422 | 0.9065 |
| 5 | **K-Neighbors Classifier** | 88.50% | 0.9101 | 0.9586 | 0.9337 | 0.8589 |

---

## 📂 6. Directory Structure

```
student-performance-predictor/
│
├── artifacts/                     # Serialized preprocessors & ML models
│   ├── best_model.joblib          # Champion Regression Model (Voting Ensemble)
│   ├── best_classifier.joblib     # Champion Risk Classifier (SVC)
│   ├── preprocessor.joblib        # Scikit-Learn ColumnTransformer pipeline
│   ├── feature_importance.csv     # XAI Permutation Importance Table
│   ├── model_metrics.csv          # Regression metrics table
│   ├── classifier_metrics.csv     # Classification metrics table
│   ├── hyperparameter_tuning_results.csv # 5-Fold CV Tuning parameters
│   └── models/                    # All individual checkpoints
│
├── data/
│   ├── StudentsPerformance.csv    # 1,000 student raw dataset
│   └── processed/                 # ML-ready train/test numpy matrices
│
├── plots/                         # 11 High-resolution visual charts
│   ├── 01_score_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_test_prep_impact.png
│   ├── 04_parental_education_impact.png
│   ├── 05_lunch_impact.png
│   ├── 06_model_performance_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_confusion_matrix.png
│   ├── 09_roc_auc_curve.png
│   ├── 10_global_feature_importance.png  # XAI Global Importance
│   └── 11_shap_directional_impact.png    # XAI Directional Attributions
│
├── src/                           # Pipeline source code
│   ├── generate_data.py           # Dataset generator
│   ├── eda.py                     # Exploratory Data Analysis & visualizer
│   ├── preprocessing.py           # Feature engineering & train-test split
│   ├── train_models.py            # Baseline model training
│   ├── hyperparameter_tuning.py   # 5-Fold CV GridSearchCV & Ensemble builder
│   ├── train_classifier.py        # Classification & Risk model suite
│   ├── goal_simulator.py          # 'What-If' Academic Goal Simulator
│   ├── explainability.py          # Explainable AI (XAI) & SHAP attributions
│   ├── evaluate_models.py         # Testing, metric computation & leaderboard
│   ├── pdf_generator.py           # ReportLab PDF Report Card generator
│   └── test_inference.py          # End-to-end dual inference verification
│
├── app.py                         # Interactive Multi-Tab Streamlit Application
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 🚀 7. Installation & How to Run

```bash
# 1. Clone Repository
git clone https://github.com/mujahith9025/Student-Performance-Predictor.git
cd Student-Performance-Predictor

# 2. Install Dependencies
python -m pip install -r requirements.txt

# 3. Launch Web App
python -m streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows).*

---
*Created as part of the Advanced Machine Learning Portfolio.*
