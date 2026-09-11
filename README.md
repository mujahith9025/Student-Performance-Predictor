# 🎓 Student Performance & Dropout Risk Predictor (Dual-Engine ML + XAI + Batch Processing)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Reports-ReportLab_5.0-red?logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)
[![XAI](https://img.shields.io/badge/Explainable_AI-SHAP_Attributions-purple)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning system optimized with **24-Feature Domain Synergy Engineering**, **Robust Outlier Scaling**, and **Super-Stacking Meta-Ensembles** to predict student examination marks, classify **Academic Pass / Fail & Dropout Risk**, process **Entire Classroom Batches via Bulk CSV Upload**, generate **Verified PDF Report Cards**, explain predictions via **Explainable AI (XAI)**, and reverse-engineer academic target pathways via a **"What-If" Academic Goal Simulator**.

---

## 📌 1. Multi-Engine Capabilities Summary

1. **Task A (Continuous Score Regression):** Predicts exact mathematical marks ($0 - 100$) using **Super-Stacking Meta-Ensembles** & **Optimized ElasticNet** (**$76.31\%$ $R^2$ accuracy**, $\pm 5.96$ marks MAE).
2. **Task B (Classification & Risk Alert):** Forecasts **Pass / Fail & Early Dropout Risk** using **Support Vector Classifiers** (**$90.00\%$ Accuracy, $0.9320$ ROC-AUC score**).
3. **Task C (Classroom Batch CSV Processing):** Upload cohort CSV files (50–500+ students) with instant pass rate statistics, grade distributions, and downloadable enriched CSVs.
4. **Task D ('What-If' Goal Simulator):** Reverse-engineers the minimum required reading and writing milestone scores and study interventions to achieve any target grade.
5. **Task E (Explainable AI):** Provides **Global Permutation Feature Importance (50 Shuffles)** and **Local Real-Time SHAP Attributions** explaining exact points added or deducted.
6. **Task F (Verified Reporting):** Generates and downloads verified, styled **PDF Academic Performance Reports** in one click.

---

## ⚡ 2. Advanced Feature Engineering & Performance Optimization

To maximize accuracy and minimize error, the pipeline expands the original 7 features into **24 high-signal domain features**:

* **Verbal Domain Synergies:**
  * `verbal_average`: Mean performance between reading and writing.
  * `verbal_differential`: Reading vs. writing divergence ($R - W$).
  * `verbal_synergy`: Geometric mean ($\sqrt{R \times W}$) capturing cross-subject proficiency.
  * `verbal_ratio`: Non-linear ratio capturing domain balance.
* **Curvature Modeling:** Quadratic power transformations (`reading_squared`, `writing_squared`) capturing performance acceleration.
* **Ordinal Socioeconomic Mapping:**
  * `parental_edu_rank`: Ordered educational gradient ($1-6$).
  * `socio_readiness_index`: Composite metric combining nutrition, prep course, and parental education.
  * Cross-interaction terms (`prep_x_reading`, `lunch_x_writing`).
* **Robust Scaling:** `RobustScaler` applied to prevent leverage distortion from extreme test scores.

---

## 📂 3. Classroom Batch Prediction (Bulk CSV Upload)

Teachers and administrators can evaluate entire classrooms in one click:
* **One-Click Template Download:** Download a pre-formatted 10-student CSV template.
* **Bulk Upload & Inference:** Upload CSV rosters of any size.
* **Classroom Overview Metrics:**
  * Total Students Processed
  * Class Average Math Score & Overall Average
  * Class Pass Rate (%)
  * Count of At-Risk / Remedial Students & Distinction Earners
* **Cohort Visualizations:** Interactive letter grade distributions and risk tier breakdowns.
* **One-Click Export:** Download the fully predicted and graded classroom spreadsheet (.CSV).

---

## 🎯 4. 'What-If' Academic Goal Simulator

The **Goal Simulator** solves the reverse optimization problem:
* **Student Target Input:** Desired Math Score (e.g. $85$ marks) or Target Grade (Grade A/A+).
* **Current Baseline:** Current reading ($65$) and writing ($62$) scores.
* **Simulator Output Roadmap:**
  * **Score Gap to Bridge:** $+22.9$ marks.
  * **Step 1:** Complete Test Preparation Course (delivers $+9.4$ marks statistical boost).
  * **Step 2:** Milestone Target: Reading Score $\rightarrow 90 / 100$ & Writing Score $\rightarrow 87 / 100$.
  * **Feasibility Rating:** Classified as *Easily Achievable*, *Moderate Effort*, or *Intensive Intervention*.

---

## 🔍 5. Explainable AI (XAI) & Global Importance Rankings

| Rank | Feature | Relative Impact | Key Finding |
| :---: | :--- | :---: | :--- |
| 🥇 | **Gender Baseline** | **40.34%** | Baseline domain variance across mathematical and verbal aptitude. |
| 🥈 | **Writing Score** | **15.60%** | Analytical writing directly influences mathematical structuring. |
| 🥉 | **Verbal Average** | **15.45%** | Cross-domain linguistic and comprehension foundation. |
| 4 | **Verbal Synergy** | **15.10%** | Geometric co-dependency between reading and writing marks. |
| 5 | **Reading Score** | **11.85%** | Core problem-comprehension capacity. |
| 6 | **Socioeconomic Readiness** | **0.78%** | Composite index of nutrition, preparation, and household degree. |

---

## 🏆 6. Model Leaderboards

### **A. Regression Leaderboard (Continuous Marks Forecast)**

| Rank | Model Name | Architecture | Test $R^2$ | Test MAE | Test RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **Optimized ElasticNet** | L1+L2 Regularized with 24 Features | **76.31%** | **$\pm$ 5.96** | **7.58** |
| 🥈 | **Optimized Ridge Regression** | $\text{L2 Shrinkage } (\alpha=10.0)$ | **76.30%** | $\pm$ 5.96 | 7.58 |
| 🥉 | **Super-Stacking Meta-Regressor** | Ridge Stacking of Huber, HistGB, RF | **76.22%** | $\pm$ 5.99 | 7.59 |
| 4 | **Optimized Huber Regressor** | Outlier-Robust Loss | **76.20%** | $\pm$ 5.98 | 7.60 |
| 5 | **Optimized HistGradientBoosting** | Histogram Gradient Boosting | **74.12%** | $\pm$ 6.33 | 7.92 |

---

### **B. Classification Leaderboard (Pass / Fail & Dropout Risk)**

| Rank | Classifier Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Optimized Support Vector Classifier (SVC)** | **90.00%** | **0.9209** | **0.9645** | **0.9422** | **0.9320** |
| 🥈 | **Optimized Logistic Regression** | 89.00% | 0.9249 | 0.9467 | 0.9357 | 0.9320 |
| 🥉 | **Optimized Gradient Boosting** | 89.00% | 0.9249 | 0.9467 | 0.9357 | 0.9234 |
| 4 | **Optimized Random Forest** | 89.50% | 0.9205 | 0.9586 | 0.9391 | 0.9126 |

---

## 📂 7. Directory Structure

```
student-performance-predictor/
│
├── artifacts/                     # Serialized preprocessors & ML models
│   ├── best_model.joblib          # Champion Regression Model (Optimized ElasticNet)
│   ├── best_classifier.joblib     # Champion Risk Classifier (Optimized SVC)
│   ├── preprocessor.joblib        # 24-Feature Robust Preprocessor
│   ├── feature_importance.csv     # XAI Permutation Importance Table
│   ├── model_metrics.csv          # Regression metrics table
│   ├── classifier_metrics.csv     # Classification metrics table
│   └── models/                    # 29 Trained model checkpoints
│
├── data/
│   ├── StudentsPerformance.csv    # 1,000 student raw dataset
│   └── processed/                 # ML-ready 24-feature train/test numpy matrices
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
│   ├── advanced_feature_engineering.py # 24-Feature Synergy & Robust Preprocessing
│   ├── optimize_models.py         # Advanced Stacking, Boosting & Super-Ensemble Training
│   ├── batch_predictor.py         # Classroom Batch CSV processing engine
│   ├── goal_simulator.py          # 'What-If' Academic Goal Simulator
│   ├── explainability.py          # Explainable AI (XAI) & SHAP attributions
│   ├── pdf_generator.py           # ReportLab PDF Report Card generator
│   ├── verify_all.py              # System-wide automated verification suite
│   └── test_inference.py          # Dual-task inference testing
│
├── app.py                         # Interactive 8-Tab Streamlit Web Application
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 🚀 8. Installation & How to Run

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

