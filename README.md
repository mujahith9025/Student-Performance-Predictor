# 🎓 Student Performance & Academic Growth Predictor

[![Live Web App](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Reports-ReportLab_5.0-red?logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)
[![Plotly](https://img.shields.io/badge/Visualizations-Plotly-239120?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> ### 🌐 Live Web Application
> Test the interactive dashboard live directly in your browser:  
> 🔗 **[https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)**

---

## 🌟 What is this Project?

**EduPredict AI** is a simple, smart, and accurate AI web app designed for students, teachers, parents, and school counselors. It analyzes **18 key student habits and academic dimensions** (such as study hours, prior exam scores, screen time, attendance, and study methods) to:

1. **🎯 Predict Exam Marks & Passing Likelihood** with **95.8% accuracy** ($\pm 2.5$ marks average error).
2. **🛡️ Provide 95% Certainty Bounds** (so you know the safe upper and lower score range).
3. **🛠️ Generate a Personalized Study Action Plan** with weekly study hour timetables and customized tips.
4. **🔄 Simulate Habit Improvements** (see how much marks increase when you add 2 extra hours of study or reduce screen time).
5. **🗺️ Reverse-Engineer Dream Targets** (set a goal like 85 or 95 marks, and get the exact study quest to reach it).
6. **📂 Analyze Whole Classrooms in Bulk** (upload a roster CSV to see class averages, risk alerts, and student groups).
7. **📄 Download Certified PDF Report Cards** (single student certificates and whole-class executive dossiers).

---

## 💡 Quick 3-Step Beginner Guide

```
Step 1: 📝 Enter or choose a student profile (or click a 1-click preset button).
Step 2: ⚡ Click "Predict Student Score" to see expected marks, letter grade, and passing chance.
Step 3: 🛠️ Follow the tailored study plan or click "Download PDF Report Card".
```

---

## ⚡ 18 Student Input Dimensions

The AI models evaluate **18 balanced student factors**:

| Category | Factors Collected | Plain-English Explanation |
| :--- | :--- | :--- |
| **📚 Academics & Literacy** | `previous_term_score`, `reading_score`, `writing_score`, `past_failures` | Previous exam baseline, reading test score, writing ability, past backlogs |
| **⏱️ Daily Study Habits** | `weekly_study_hours`, `attendance_rate`, `study_method` | Self-study time, class attendance %, study technique (Active Recall, Spaced Repetition, Group) |
| **📱 Lifestyle & Balance** | `daily_screen_time_hours`, `sleep_hours_per_day` | Phone/social media screen time, daily sleep duration |
| **👥 Mentorship & Support** | `parental_involvement`, `tutoring_support`, `test_preparation_course` | Home guidance, extra tutoring sessions, exam prep course status |
| **👤 Demographics & Environment** | `parental_level_of_education`, `lunch`, `internet_access`, `extracurricular_activities`, `gender`, `race/ethnicity` | Home learning environment, meal support, home internet, sports/clubs |

---

## 🏆 Model Accuracy & Benchmarks

### **A. Exam Marks Prediction ($0 - 100$)**
| Rank | Model Name | Architecture | Test $R^2$ Accuracy | Average Error (MAE) |
| :---: | :--- | :--- | :---: | :---: |
| 🥇 | **Super-Stacking Meta-Regressor** | Ridge Stacking of Huber, HistGB & Random Forest | **95.75%** | **$\pm$ 2.54 marks** |
| 🥈 | **Optimized ElasticNet** | L1+L2 Regularized with 54 features | **95.74%** | $\pm$ 2.54 marks |
| 🥉 | **Optimized Ridge Regression** | L2 Shrinkage Regularization | **95.68%** | $\pm$ 2.56 marks |
| 4 | **Optimized HistGradientBoosting** | Histogram Gradient Boosting Trees | **95.22%** | $\pm$ 2.68 marks |
| 5 | **Optimized Random Forest** | Multi-Tree Ensemble (150 trees) | **94.88%** | $\pm$ 2.76 marks |

---

### **B. Passing Likelihood & Risk Classifier ($50+$ marks)**
| Rank | Classifier Name | Accuracy | Precision | Recall | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting Classifier** | **98.00%** | **98.68%** | **98.94%** | **0.9950** |
| 🥈 | **Optimized Random Forest Classifier** | **97.75%** | **98.42%** | **98.94%** | **0.9942** |
| 🥉 | **Support Vector Classifier (SVC)** | **97.50%** | **98.15%** | **98.94%** | **0.9928** |

---

## 📱 6 Interactive Dashboard Workspaces

1. **🎯 Student Predictor:** 1-click presets, customizable habit sliders, 4-tile bento summary, and 1-click PDF download.
2. **📂 Classroom Analytics:** Upload custom roster CSV or load sample classes (50, 100, 200 students) with automatic intervention group clusters.
3. **🔄 Growth Simulator:** Real-time Before vs After slider comparisons showing the exact mark boost from habit changes.
4. **🗺️ Goal Planner:** Pick Bronze (50), Silver (70), Gold (85), or Diamond (95) to reverse-engineer the required study quests.
5. **🔍 What Drives Scores?:** Visual factor rankings and dataset charts explaining what habits help or lower scores the most.
6. **⚙️ AI Models & Accuracy:** Full accuracy leaderboards, 95% confidence intervals, and architecture flowcharts.

---

## 📂 Project Structure

```
student-performance-predictor/
├── app.py                         # Clean, simple & user-friendly Streamlit web app
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies list
├── README.md                      # Project documentation
│
├── artifacts/                     # Trained ML models and preprocessors
│   ├── best_model.joblib          # Champion Super-Stacking Meta-Regressor (95.8% R²)
│   ├── best_classifier.joblib     # Champion Risk Classifier (98.0% Accuracy)
│   ├── preprocessor.joblib        # 54-column feature preprocessor
│   ├── uncertainty_model.joblib   # 95% Conformal Prediction uncertainty margins
│   ├── archetype_clusterer.joblib # 4-Cluster K-Means Learning Persona model
│   └── models/                    # All individual trained model checkpoints
│
├── data/
│   ├── StudentsPerformance.csv    # 2,000-student benchmark dataset
│   └── sample_classrooms/         # Pre-built benchmark classroom CSV files
│
├── plots/                         # High-resolution visual charts
│
└── src/                           # Backend Python engines
    ├── advanced_feature_engineering.py # 43 interaction & synergy metrics
    ├── advanced_ml_statistical.py      # Conformal prediction & K-Means clustering
    ├── batch_predictor.py              # Bulk classroom CSV processing engine
    ├── goal_simulator.py               # 'What-If' Academic Goal Simulator
    ├── explainability.py               # Feature importance & point drivers
    ├── prescriptive_solutions.py       # Personalized study action plan engine
    ├── pdf_generator.py                # Official certified PDF report generator
    ├── plotly_charts.py                # Interactive charts
    ├── sample_generator.py             # Synthetic classroom cohort generator
    └── verify_all.py                   # Automated 10-engine verification test
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/mujahith9025/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
*(On Windows, you can simply double-click **`run_app.bat`**).*

---

## 🌐 Live Streamlit Cloud Deployment

Visit the live app anytime at:  
👉 **[https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/](https://student-performance-predictor-k475y5vl7pvyx99qeabrtu.streamlit.app/)**
