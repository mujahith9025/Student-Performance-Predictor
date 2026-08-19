# 🎓 Student Academic Performance Predictor

[![CI/CD Testing Pipeline](https://github.com/mujahith9025/student-Performance-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/mujahith9025/student-Performance-Predictor/actions/workflows/ci.yml)
[![FastAPI Microservice](https://img.shields.io/badge/FastAPI-v2.0-009688.svg?logo=fastapi&logoColor=white)](https://github.com/mujahith9025/student-Performance-Predictor)
[![Docker Container](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://github.com/mujahith9025/student-Performance-Predictor)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-predictor-7fzsxbt6zeodyepgp7rqjy.streamlit.app/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Automated Tests](https://img.shields.io/badge/Tests-29%20Passed%20%2F%20100%25-success.svg)](https://github.com/mujahith9025/student-Performance-Predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🌐 **Live Demo:** Try the online web app directly in your browser:  
> 👉 **[Launch Student Performance Predictor on Streamlit Cloud](https://student-performance-predictor-7fzsxbt6zeodyepgp7rqjy.streamlit.app/)**

An end-to-end Machine Learning project and interactive **Streamlit web application + FastAPI REST API Microservice** that predicts a student's academic performance index based on study habits, previous exam scores, sleep patterns, practice tests, and extracurricular activities.

Built using **10,000 real student records from Kaggle** with a modular scikit-learn training pipeline ($R^2 \approx 98.84\%$).

---

## 📌 Features & Highlights

- **⚡ REST API Microservice (FastAPI)**:
  - High-performance asynchronous REST API powered by **FastAPI & Uvicorn**.
  - Interactive **Swagger UI (`/docs`)** and **ReDoc (`/redoc`)** with full Pydantic data validation schemas.
  - Endpoints for single prediction, batch processing, SHAP explainability, reverse goal optimization, database history queries, and PDF binary streaming.
  - Ready for integration with mobile apps, LMS platforms, or external web services.
- **🐳 Docker Containerization & Docker Compose**:
  - Fully containerized production environment with multi-stage `python:3.11-slim` image.
  - Dual service orchestration running Streamlit UI (Port `8501`) and FastAPI (Port `8000`).
  - Automated healthcheck endpoints (`/_stcore/health` and `/health`).
  - Persistent volume binding for SQLite database (`./data:/app/data`).
- **🚀 Automated CI/CD Testing Pipeline (GitHub Actions)**:
  - Multi-OS matrix testing on **Ubuntu** and **Windows**.
  - Multi-Python compatibility testing across **Python 3.10, 3.11, and 3.12**.
  - **29 automated unit & integration tests** covering API endpoints, data loaders, feature engineering, model inference, SHAP explainability, reverse goal optimization, PDF generation, and SQLite CRUD.
  - Automated Docker container build verification on GitHub Actions.
- **🗄️ SQLite / Supabase Database Persistence Layer**:
  - **Persistent Prediction Audit Trail:** Automatically persists every predicted student record, confidence interval, 6-D habit balance index, teacher counseling remarks, and risk flags to an embedded SQLite database (`data/student_records.db`).
  - **📈 Longitudinal Student Score Growth Timeline:** Interactive time-series progress charts plotting a student's score evolution and habit changes across multiple tests/semesters.
  - **🎯 Target Roadmap Bookmarks:** Preserves Reverse Goal Solver pathways and weekly study commitments.
  - **📥 Database Export Center:** 1-click export of complete historical audit logs into **CSV** and **JSON** formats.
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
- **🔮 Single Student Predictor**: Interactive sliders and dropdowns with real-time score prediction, letter grades (A+, A, B, C, D, F), academic standing indicators, and personalized study boost recommendations.
- **🎯 Reverse Goal Simulator ("Target Score Solver")**:
  - Solves the **inverse academic optimization problem**: input a desired target score (e.g., `85` or `95`), and the system calculates the exact study habits required to achieve it.
  - Automatically generates **3 tailored pathways** (Balanced Mastery, Mock-Test Sprint, Express Efficiency).
- **📄 Official Academic Diagnostic PDF Report Generator**:
  - Automatically compiles and exports a clean vector-rendered **PDF Academic Diagnostic Report** in 1 click.
- **🕸️ Student Habit Radar Chart**:
  - Multi-dimensional polar/spider chart comparing the student's **6 key study & health dimensions** against **🌟 A+ Top Performers** and **👥 Cohort Averages**.
- **🧠 SHAP Explainability (Explainable AI - XAI)**:
  - Local waterfall attribution and global game-theoretic Shapley value rankings.
- **🎯 Prediction Confidence Intervals (80%–99%)**:
  - Rigorous uncertainty quantification with calibrated prediction bounds $[y_{\text{lower}}, y_{\text{upper}}]$.

---

## ⚡ REST API Endpoints & Swagger Docs

The FastAPI microservice runs at **`http://localhost:8000`** with interactive Swagger documentation at **`http://localhost:8000/docs`**:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status, loaded models, and database count |
| `GET` | `/api/v1/meta/subjects` | Metadata for all 4 academic subject disciplines |
| `GET` | `/api/v1/meta/models` | Available ML algorithms and performance benchmarks |
| `POST` | `/api/v1/predict` | Single student prediction with confidence bounds & risk flags |
| `POST` | `/api/v1/predict/batch`| Batch student predictions with cohort pass rates |
| `POST` | `/api/v1/explain` | SHAP waterfall decision attribution & feature rankings |
| `POST` | `/api/v1/solve-goal` | Reverse Goal Simulator (optimal study habit pathways) |
| `GET` | `/api/v1/history` | Query database prediction audit records |
| `GET` | `/api/v1/history/timeline/{student_name}` | Longitudinal score growth progression for a student |
| `GET` | `/api/v1/stats` | High-level database summary metrics |
| `POST` | `/api/v1/report/pdf` | Generate & stream vector-rendered academic diagnostic PDF |

### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "student_name": "Alex Johnson",
       "hours_studied": 6.0,
       "previous_score": 80.0,
       "sleep_hours": 7.5,
       "sample_papers": 4,
       "extracurricular": "Yes",
       "subject": "Mathematics & Statistics",
       "confidence_level": "95%",
       "save_to_db": true
     }'
```

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
│   ├── test_api.py               # Unit & integration tests for FastAPI REST API
│   ├── test_data_loader.py       # Unit tests for data loading & feature engineering
│   ├── test_models.py            # Unit tests for 9 models & inference
│   ├── test_explainability.py    # Unit tests for SHAP waterfall attributions
│   ├── test_goal_solver.py       # Unit tests for reverse goal solver
│   ├── test_pdf_generator.py     # Unit tests for PDF compilation
│   ├── test_db.py                # Unit tests for SQLite database persistence
│   └── test_app_integration.py   # Integration tests for Streamlit app helpers
├── api.py                        # FastAPI REST API Microservice (Port 8000)
├── app.py                        # Interactive Streamlit Web Application (Port 8501)
├── Dockerfile                    # Multi-stage production container build
├── docker-compose.yml            # Dual-service Docker Compose orchestration
├── .dockerignore                 # Excluded build context patterns
├── pytest.ini                    # Pytest configuration settings
├── requirements.txt              # Project dependencies list
└── README.md                     # Documentation & online deployment guide
```

---

## 🐳 Running with Docker

You can run both the Streamlit UI and the FastAPI REST API microservice simultaneously:

```bash
# Start both Streamlit (8501) and FastAPI (8000)
docker-compose up -d
```
- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🚀 Local Quickstart & Testing

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite (29 Tests)
```powershell
pytest tests/
```

### 3. Launch Streamlit Web UI
```powershell
streamlit run app.py
```

### 4. Launch FastAPI REST API Microservice
```powershell
python api.py
# or: uvicorn api:app --reload --port 8000
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
