# 🎓 Student Performance Predictor (Machine Learning Project)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end beginner-friendly Machine Learning project designed to predict student examination marks (Mathematics) based on demographic factors, socio-economic indicators, study habits, and existing subject competencies.

---

## 📌 1. Project Overview & Problem Statement
Understanding the factors that influence academic success allows educators and parents to identify at-risk students early and provide targeted academic interventions. 

This project implements a complete **Supervised Machine Learning Regression Pipeline**:
* **Input Features:** Gender, Ethnicity, Parental Level of Education, Lunch Type, Test Preparation Course, Reading Score, Writing Score.
* **Target Variable:** Mathematics Examination Marks ($0 - 100$).
* **Best Model:** **Ridge Regression** achieving **$76.1\%$ $R^2$ accuracy** and an average error of **$\pm 6.02$ marks**.

---

## 🏗️ 2. Project Architecture & Phases

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Phase 1: Dataset Setup │ ──► │     Phase 2: EDA        │ ──► │  Phase 3: Preprocessing │
│ (1,000 Student Records) │     │ (Distributions & Corrs) │     │ (ColumnTransformer OHE) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐                  │
│    Phase 6: Web App     │ ◄── │   Phase 5: Evaluation   │ ◄────────────────┘
│ (Streamlit Interactive) │     │ (Leaderboard & Metrics) │     Phase 4: Model Training
└─────────────────────────┘     └─────────────────────────┘     (6 ML Algorithms Trained)
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

## 🏆 4. Model Comparison & Results

We evaluated 6 regression algorithms on 200 unseen test students:

| Rank | Model Name | Train $R^2$ | Test $R^2$ | Test MAE *(Avg Error)* | Test RMSE | Overfitting Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Ridge Regression** | **0.7635** | **0.7606 (76.1%)** | **$\pm$ 6.02 marks** | **7.62** | ✅ Optimal fit (No Overfitting) |
| 🥈 | **Lasso Regression** | 0.7611 | 0.7604 (76.0%) | $\pm$ 5.99 marks | 7.62 | ✅ High Generalization |
| 🥉 | **Linear Regression** | 0.7635 | 0.7602 (76.0%) | $\pm$ 6.03 marks | 7.63 | ✅ Strong Baseline |
| 4 | **Random Forest Regressor** | 0.8919 | 0.7305 (73.1%) | $\pm$ 6.41 marks | 8.09 | ⚠️ Slight Overfitting |
| 5 | **K-Neighbors Regressor** | 0.7835 | 0.6885 (68.9%) | $\pm$ 6.97 marks | 8.69 | ⚠️ Moderate Variance |
| 6 | **Decision Tree Regressor** | 0.8118 | 0.6508 (65.1%) | $\pm$ 7.22 marks | 9.20 | ⚠️ High Variance |

---

## 📂 5. Directory Structure

```
student-performance-predictor/
│
├── artifacts/                     # Serialized preprocessors & ML models
│   ├── best_model.joblib          # Champion trained model (Ridge Regression)
│   ├── preprocessor.joblib        # Scikit-Learn ColumnTransformer pipeline
│   ├── model_metrics.csv          # Evaluation metrics across all models
│   └── models/                    # Individual checkpoints for all 6 models
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
│   ├── train_models.py            # Model training & checkpointing
│   ├── evaluate_models.py         # Testing, metric computation & ranking
│   └── test_inference.py          # End-to-end inference verification
│
├── app.py                         # Interactive Streamlit Web Application
├── run_app.bat                    # 1-click Windows launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project Documentation
```

---

## 🚀 6. Installation & How to Run

### Step 1: Clone or Navigate to Project Directory
```bash
cd "d:\My Project\student-performance-predictor"
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run Machine Learning Pipeline (Optional / Step-by-Step)
```bash
# 1. Run EDA & generate charts
python src/eda.py

# 2. Run Data Preprocessing
python src/preprocessing.py

# 3. Train all ML models
python src/train_models.py

# 4. Evaluate models and select champion
python src/evaluate_models.py
```

### Step 4: Launch Interactive Web App
```bash
python -m streamlit run app.py
```
*(Or simply double-click `run_app.bat` on Windows).*

---

## 💡 7. Key Takeaways & Future Enhancements
* **Key Takeaway:** Linear models with $L_2$ regularization (Ridge) outperform complex decision trees on this problem due to high multicollinearity between academic subject scores.
* **Future Work:**
  * Add study hour time tracking as an additional feature.
  * Integrate SMS / Email notification alerts for at-risk students.
  * Deploy the application to Streamlit Community Cloud / HuggingFace Spaces.

---
*Created as part of the Beginner-to-Advanced Machine Learning Portfolio.*
