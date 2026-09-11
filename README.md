# 🎓 Student Performance & Dropout Risk Predictor (14-Feature Multi-Dimensional Dual ML)

[![Live Web App](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Reports-ReportLab_5.0-red?logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)
[![Plotly](https://img.shields.io/badge/Visualizations-Plotly-239120?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> ### 🌐 Live Web Application
> Experience the full interactive dashboard live in your browser:  
> 🔗 **[https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)**

---

An end-to-end Machine Learning system engineered with **14-Feature Multi-Dimensional Educational & Behavioral Profiling** (38 Total Synergy Features), **Robust Outlier Normalization**, and **Dual-Task AI Models** to forecast student examination marks, classify **Academic Pass / Fail & Dropout Risk**, process **Entire Classroom Batches via Bulk CSV Upload**, generate **Verified PDF Performance Certificates**, explain predictions via **Explainable AI (SHAP & Permutation Importance)**, and reverse-engineer milestone pathways via a **"What-If" Gamified Goal Simulator**.

---

## 📌 1. Multi-Engine Capabilities Summary

1. **Task A (Continuous Score Forecasting):** Predicts exact mathematical marks ($0 - 100$) using **Optimized ElasticNet & Super-Stacking Meta-Ensembles** (**$90.07\%$ $R^2$ accuracy**, $\pm 3.47$ marks MAE).
2. **Task B (Classification & Risk Alert):** Forecasts **Pass / Fail & Early Dropout Risk** using **Support Vector Classifiers** (**$97.00\%$ Accuracy, $0.9853$ ROC-AUC score**).
3. **Task C (AI Prescriptive Solution & Intervention Engine):** Clinical diagnostic algorithm predicting personalized study plans, weekly subject study hour allocations, milestone timelines, and projected score uplifts.
4. **Task D (Classroom Batch CSV Hub & Synthetic Generator):** 5 pre-built benchmark classroom CSV files + dynamic generator for custom cohort sizes (10 to 500 students) with AI clustered intervention cohorts.
5. **Task E ('What-If' Goal Simulator):** Reverse-engineers the minimum required reading and writing milestone scores, study hours, and attendance habits to achieve any target grade.
6. **Task F (Explainable AI):** Provides **Global Permutation Feature Importance (50 Shuffles)** and **Local Real-Time SHAP Attributions** explaining exact points added or deducted.
7. **Task G (Verified Reporting):** Generates and downloads verified, styled **PDF Academic Performance Reports** (Single Student & Whole Classroom Executive Summaries).

---

## ⚡ 2. 14-Feature Multi-Dimensional Profile & Advanced Engineering

The system collects **14 comprehensive student attributes across 4 core educational pillars**:

* **📚 Academic Literacy & Deliberate Study:**
  * `reading score` (0 - 100): Reading comprehension and textual literacy.
  * `writing score` (0 - 100): Written synthesis and problem structure.
  * `attendance_rate` (50% - 100%): Classroom attendance continuity and lecture engagement.
  * `weekly_study_hours` (1 - 40 hrs): Self-study and deliberate homework practice.
  * `past_failures` (0 - 4): Prior subject backlogs and historical concept fragility.
* **👤 Demographics, Household & Digital Capital:**
  * `gender`: Demographic baseline profile.
  * `race/ethnicity`: Background student cohort.
  * `parental level of education`: Household academic attainment gradient.
  * `lunch`: Nutritional security plan (standard / free/reduced).
  * `internet_access`: Home high-speed digital connection for online portals.
* **🌱 Lifestyle, Curricular & Support Services:**
  * `sleep_hours_per_day` (4.0 - 10.0 hrs): Daily sleep balance and cognitive fatigue avoidance.
  * `test preparation course`: Standardized exam preparation completion.
  * `extracurricular_activities`: Balanced extracurricular participation.
  * `tutoring_support`: Active tutoring assistance (none, peer tutoring, private tutor).

From these 14 raw inputs, the pipeline engineers **38 domain synergy, interaction, curvature, and behavioral indices**:
* `verbal_average`, `verbal_differential`, `verbal_synergy`, `verbal_ratio`
* `reading_squared`, `writing_squared` (quadratic power curves)
* `study_attendance_synergy`, `academic_effort_index`
* `academic_risk_friction`, `wellness_lifestyle_score`, `socio_readiness_index`
* Cross-interaction terms (`prep_x_reading`, `lunch_x_writing`, `study_x_attendance`, `tutor_x_study`).

---

## 🏆 3. Model Leaderboards

### **A. Regression Leaderboard (Continuous Marks Forecast)**

| Rank | Model Name | Architecture | Test $R^2$ | Test MAE | Test RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **Optimized ElasticNet** | L1+L2 Regularized with 38 Features | **90.07%** | **$\pm$ 3.47** | **4.36** |
| 🥈 | **Super-Stacking Meta-Regressor** | Ridge Stacking of Huber, HistGB, RF | **90.06%** | $\pm$ 3.47 | 4.36 |
| 🥉 | **Optimized Ridge Regression** | $\text{L2 Shrinkage } (\alpha=10.0)$ | **89.94%** | $\pm$ 3.51 | 4.38 |
| 4 | **Optimized Huber Regressor** | Outlier-Robust Loss | **89.84%** | $\pm$ 3.53 | 4.41 |
| 5 | **Optimized HistGradientBoosting** | Histogram Gradient Boosting | **89.45%** | $\pm$ 3.55 | 4.49 |

---

### **B. Classification Leaderboard (Pass / Fail & Dropout Risk)**

| Rank | Classifier Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Support Vector Classifier (SVC)** | **97.00%** | **98.12%** | **98.41%** | **0.9826** | **0.9853** |
| 🥈 | **Logistic Regression (Tuned)** | **97.25%** | **98.15%** | **98.68%** | **0.9841** | **0.9849** |
| 🥉 | **Gradient Boosting Classifier** | **96.75%** | **97.89%** | **98.41%** | **0.9815** | **0.9825** |

---

## 📂 4. Directory Structure

```
student-performance-predictor/
│
├── artifacts/                     # Serialized preprocessors & ML models
│   ├── best_model.joblib          # Champion Regression Model (Optimized ElasticNet)
│   ├── best_classifier.joblib     # Champion Risk Classifier (Optimized SVC)
│   ├── preprocessor.joblib        # 38-Feature Robust Preprocessor
│   ├── feature_importance.csv     # XAI Permutation Importance Table
│   ├── model_metrics.csv          # Regression metrics table
│   ├── classifier_metrics.csv     # Classification metrics table
│   └── models/                    # Trained model checkpoints
│
├── data/
│   ├── StudentsPerformance.csv    # Benchmark dataset
│   ├── sample_classrooms/         # 5 Pre-built bulk CSV datasets
│   └── processed/                 # ML-ready train/test numpy matrices
│
├── plots/                         # High-resolution visual charts
│
├── src/                           # Pipeline source code
│   ├── advanced_feature_engineering.py # 38-Feature Synergy & Robust Preprocessing
│   ├── optimize_models.py         # Advanced Stacking, Boosting & Super-Ensemble Training
│   ├── train_classifier.py        # SVC & Ensemble Risk Classifier Pipeline
│   ├── batch_predictor.py         # Classroom Batch CSV processing engine
│   ├── goal_simulator.py          # 'What-If' Academic Goal Simulator
│   ├── explainability.py          # Explainable AI (XAI) & SHAP attributions
│   ├── prescriptive_solutions.py  # Clinical diagnostic & action planning engine
│   ├── sample_generator.py        # Dynamic synthetic classroom generator
│   ├── plotly_charts.py           # Interactive Plotly chart builders
│   ├── pdf_generator.py           # ReportLab PDF Report Card generator
│   └── verify_all.py              # System-wide automated verification suite
│
├── app.py                         # Streamlined 5-Tab Bento Grid Streamlit Web App
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 🚀 5. Installation & How to Run Locally

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

## 🌐 6. Live Streamlit Cloud Deployment

Access the live cloud deployment anytime at:  
👉 **[https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)**
