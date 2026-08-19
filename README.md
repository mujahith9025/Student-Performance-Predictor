# 🎓 Student Academic Performance Predictor

[![CI/CD Testing Pipeline](https://github.com/mujahith9025/student-Performance-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/mujahith9025/student-Performance-Predictor/actions/workflows/ci.yml)
[![Docker Container](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://github.com/mujahith9025/student-Performance-Predictor)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-predictor-7fzsxbt6zeodyepgp7rqjy.streamlit.app/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Automated Tests](https://img.shields.io/badge/Tests-19%20Passed%20%2F%20100%25-success.svg)](https://github.com/mujahith9025/student-Performance-Predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🌐 **Live Demo:** Try the online web app directly in your browser:  
> 👉 **[Launch Student Performance Predictor on Streamlit Cloud](https://student-performance-predictor-7fzsxbt6zeodyepgp7rqjy.streamlit.app/)**

An end-to-end Machine Learning project and interactive **Streamlit web application** that predicts a student's academic performance index based on study habits, previous exam scores, sleep patterns, practice tests, and extracurricular activities.

Built using **10,000 real student records from Kaggle** with a modular scikit-learn training pipeline ($R^2 \approx 98.84\%$).

---

## 📌 Features & Highlights

- **🐳 Docker Containerization & Docker Compose**:
  - Fully containerized production environment with multi-stage `python:3.11-slim` image.
  - Automated healthcheck endpoint at `http://localhost:8501/_stcore/health`.
  - Persistent volume binding for SQLite database (`./data:/app/data`).
  - Single-command orchestration via `docker-compose up -d`.
- **🚀 Automated CI/CD Testing Pipeline (GitHub Actions)**:
  - Multi-OS matrix testing on **Ubuntu** and **Windows**.
  - Multi-Python compatibility testing across **Python 3.10, 3.11, and 3.12**.
  - **19 automated unit & integration tests** covering data loaders, feature engineering, model inference, SHAP explainability, reverse goal optimization, PDF generation, and SQLite CRUD.
  - Automated Docker container build verification on GitHub Actions.
- **🗄️ SQLite / Supabase Database Persistence Layer**:
  - **Persistent Prediction Audit Trail:** Automatically persists every predicted student record, confidence interval, 6-D habit balance index, teacher counseling remarks, and risk flags to an embedded SQLite database (`data/student_records.db`).
  - **📈 Longitudinal Student Score Growth Timeline:** Interactive time-series progress charts plotting a student's score evolution and habit changes across multiple tests/semesters.
  - **🎯 Target Roadmap Bookmarks:** Preserves Reverse Goal Solver pathways and weekly study commitments.
  - **📥 Database Export Center:** 1-click export of complete historical audit logs into **CSV** and **JSON** formats.
  - **Cloud Ready:** Plug-and-play architecture with zero extra configuration on SQLite, supporting Supabase PostgreSQL cloud sync via secrets.
- **🎭 Teacher vs. Student View Toggle**:
  - Interactive persona switcher adapting the application layout, clinical diagnostics, and action roadmaps:
    - 🧑‍🎓 **Student Mode:** Motivational, goal-oriented view with habit scorecard, score booster missions, and self-directed study targets.
    - 👩‍🏫 **Teacher & Counselor Mode:** Comprehensive educator interface with **Early Warning Risk Flags** (e.g. Failure Risk, Severe Sleep Deficit, Concept Gaps), Educator Action Plans, custom counseling notes, and class risk matrix breakdown.
- **📚 Multi-Subject Dataset Switcher**:
  - Seamlessly switch between **4 academic disciplines**:
    1. 🌐 **General Academics:** Cross-disciplinary baseline benchmark ($10,000$ students).
    2. 📐 **Mathematics & Statistics:** Quantitative logic emphasizing practice density and problem solving.
    3. 🔬 **Science & Physics:** Laboratory & STEM focus with cognitive rest & foundational scoring.
    4. 📖 **Humanities & Literature:** Arts, language, and extracurricular communication focus.
  - Dynamically loads discipline-specific models, tailored SHAP waterfalls, and custom subject recommendations.
- **🔮 Single Student Predictor**: Interactive sliders and dropdowns with real-time score prediction, letter grades (A+, A, B, C, D, F), academic standing indicators, and personalized study boost recommendations.
- **🎯 Reverse Goal Simulator ("Target Score Solver")**:
  - Solves the **inverse academic optimization problem**: input a desired target score (e.g., `85` or `95`), and the system calculates the exact study habits required to achieve it.
  - Automatically generates **3 tailored pathways**:
    1. ⚖️ **Balanced Mastery Pathway** (Optimal 7.5 hrs sleep with sustainable study habits)
    2. 📄 **Mock-Test Sprint Pathway** (High mock paper volume with active practice density)
    3. ⚡ **Express Efficiency Pathway** (Minimal daily study hours needed)
  - Features **Before vs. After Habit Radar Transformation** overlays, weekly study schedules, and 1-click **Target Roadmap PDF** exports.
- **📄 Official Academic Diagnostic PDF Report Generator**:
  - Automatically compiles and exports a clean vector-rendered **PDF Academic Diagnostic Report** in 1 click.
  - Formatted for student advising, parent-teacher reviews, or personal study planning.
  - Contains executive score summaries, confidence bounds, 6-D habit diagnostics, SHAP factor attribution matrix, and concrete personalized study recommendations.
- **🕸️ Student Habit Radar Chart**:
  - Multi-dimensional polar/spider chart comparing the student's **6 key study & health dimensions** (Study Time, Exam Foundation, Mock Practice, Sleep Quality, Study Effort, and Extracurriculars) against **🌟 A+ Top Performers** and **👥 Cohort Averages**.
  - Calculates a holistic **Habit Balance Index (0–100%)** with personalized qualitative assessment.
  - Grade-tier radar footprint analysis comparing habit footprints across A+, B, and F students.
- **🔬 Custom Feature Engineering**:
  - ⚖️ **Study-to-Sleep Ratio**: $\frac{\text{Hours Studied}}{\text{Sleep Hours}}$ (Work-rest balance metric).
  - 📄 **Practice Density**: $\frac{\text{Mock Papers Practiced}}{\text{Hours Studied} + 1}$ (Active practice intensity).
  - ⚡ **Study Effort Score**: $(0.6 \times \text{Hours}) + (0.4 \times \text{Mock Papers})$ (Composite academic effort index).
- **🧠 SHAP Explainability (Explainable AI - XAI)**:
  - **Local SHAP Waterfall**: Breaks down each student's predicted score starting from the cohort baseline ($\approx 54.8$ pts) and shows the exact point contributions ($+16.0$ for previous scores, $+3.9$ for study hours, etc.).
  - **Global SHAP Feature Importance**: Game-theoretic Shapley value rankings illustrating which habits influence grades the most across the entire student population.
- **🎯 Prediction Confidence Intervals (80%–99%)**:
  - Provides rigorous uncertainty quantification: instead of a single point score, generates calibrated **Prediction Intervals** $[y_{\text{lower}}, y_{\text{upper}}]$.
  - Supports user-selectable confidence levels: **80% (±2.66 pts)**, **90% (±3.41 pts)**, **95% (±4.07 pts)**, and **99% (±5.34 pts)**.
  - Features visual **Confidence Band Gauges** and empirical coverage calibration on 1,975 real test students.
- **⚡ Advanced Gradient Boosters & Ensembles**:
  - **XGBoost Regressor**: Extreme Gradient Boosting with tree regularization.
  - **LightGBM Regressor**: Fast, leaf-wise gradient boosting.
  - **HistGradientBoosting**: Binned gradient boosting for rapid convergence.
  - **Stacking Ensemble Regressor**: Meta-learner combining Linear, Random Forest, and GBDT predictions.
- **📂 Batch Prediction (CSV Upload)**: Upload class rosters or cohort CSVs to generate predictions, automatically engineer features, and export results with confidence intervals in one click.
- **📊 Multi-Model Comparison Dashboard**: Interactive comparison across 9 regression models with evaluation metrics ($R^2$, MAE, RMSE).

---

## 📂 Project Directory Structure

```
d:/Project/student_performance_predictor/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions Automated CI/CD & Docker Pipeline
├── data/
│   ├── download_dataset.py       # Downloads & caches the official Kaggle dataset
│   ├── Student_Performance.csv   # 10,000 real student records from Kaggle
│   ├── sample_batch_test.csv     # Sample batch dataset for CSV upload testing
│   └── student_records.db        # SQLite Database for persistent prediction audit logs
├── models/
│   ├── best_model.pkl            # Serialized scikit-learn Pipeline (Champion model)
│   ├── all_trained_models.pkl    # Bundle of all 9 trained models (XGBoost, LightGBM, etc.)
│   ├── subject_models.pkl        # Multi-Subject tailored models bundle
│   ├── subject_metrics.json      # Cross-subject performance metrics & benchmarks
│   └── metrics.json              # Full benchmark comparison metrics & feature weights
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading, custom feature engineering, & train-test split
│   ├── db.py                     # SQLite / Supabase Database Persistence Engine & CRUD
│   ├── explainability.py         # SHAP Tree & Linear waterfall explainers
│   ├── goal_solver.py            # Reverse Goal Simulator (Inverse optimization solver)
│   ├── pdf_generator.py          # ReportLab vector-rendered Academic PDF Report generator
│   └── train.py                  # Trains & benchmarks all 9 regression models
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py       # Unit tests for data loading & feature engineering
│   ├── test_models.py            # Unit tests for 9 models & inference
│   ├── test_explainability.py    # Unit tests for SHAP waterfall attributions
│   ├── test_goal_solver.py       # Unit tests for reverse goal solver
│   ├── test_pdf_generator.py     # Unit tests for PDF compilation
│   ├── test_db.py                # Unit tests for SQLite database persistence
│   └── test_app_integration.py   # Integration tests for Streamlit app helpers
├── app.py                        # Interactive Streamlit Web Application (6 Tabs)
├── Dockerfile                    # Multi-stage production container build
├── docker-compose.yml            # 1-command Docker Compose orchestration
├── .dockerignore                 # Excluded build context patterns
├── pytest.ini                    # Pytest configuration settings
├── requirements.txt              # Project dependencies list
└── README.md                     # Documentation & online deployment guide
```

---

## 🐳 Running with Docker

You can run the entire application in a standardized container with zero local Python setup required:

### Option A: Using Docker Compose (Recommended)
```bash
docker-compose up -d
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

To stop the container:
```bash
docker-compose down
```

### Option B: Using Plain Docker CLI
```bash
# 1. Build the Docker image
docker build -t student-performance-predictor:latest .

# 2. Run container with port forwarding and database volume mount
docker run -d -p 8501:8501 -v $(pwd)/data:/app/data --name student_predictor_app student-performance-predictor:latest
```

---

## 🚀 Local Quickstart & Testing

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```powershell
pytest tests/
```

### 3. Launch Streamlit Application
```powershell
streamlit run app.py
```

---

## 🏆 Model Benchmark Results

On the test set (1,975 real student records):

| Model | Category | Test $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Baseline | **98.84%** | **1.65** | **2.08** |
| **Ridge Regression** | Regularized Linear | **98.84%** | **1.65** | **2.08** |
| **Stacking Ensemble** | Advanced Ensemble | **98.84%** | **1.64** | **2.07** |
| **XGBoost Regressor** | Advanced Gradient Booster | **98.77%** | **1.70** | **2.14** |
| **LightGBM Regressor** | Advanced Gradient Booster | **98.77%** | **1.70** | **2.14** |
| **Gradient Boosting (GBDT)**| Advanced Gradient Booster | **98.77%** | **1.70** | **2.14** |
| **HistGradientBoosting** | Advanced Gradient Booster | **98.76%** | **1.71** | **2.15** |
| **Random Forest** | Ensemble Bagging | **98.61%** | **1.81** | **2.28** |
| **Decision Tree** | Tree Model | **97.29%** | **2.55** | **3.17** |

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
