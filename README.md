# 🎓 Student Academic Performance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end beginner-friendly Machine Learning project and interactive **Streamlit web application** that predicts a student's academic performance index based on study habits, previous exam scores, sleep patterns, practice tests, and extracurricular activities.

Built using **10,000 real student records from Kaggle** with a modular scikit-learn training pipeline ($R^2 \approx 98.84\%$).

---

## 📌 Features & Highlights

- **🔮 Single Student Predictor**: Interactive sliders and dropdowns with real-time score prediction, letter grades (A+, A, B, C, D, F), academic standing indicators, and personalized study boost recommendations.
- **📂 Batch Prediction (CSV Upload)**: Upload class rosters or cohort CSVs to generate predictions, cohort averages, pass rates, and export results in one click.
- **📊 Model Performance Leaderboard**: Benchmark comparison across 5 models (**Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting**) with evaluation metrics ($R^2$, MAE, RMSE).
- **📈 Exploratory Data Analysis (EDA)**: Interactive correlation matrix heatmaps, feature weight charts, and distribution scatter plots powered by Plotly.
- **☁️ Online Ready**: Optimized for 1-click cloud deployment on **Streamlit Community Cloud**.

---

## 📊 Dataset Overview (Kaggle)

- **Dataset Source:** [Kaggle - Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- **Observations:** 10,000 student entries.
- **Input Variables:**
  1. `Hours_Studied`: Total hours spent studying per day (1 to 9 hours).
  2. `Previous_Score`: Marks scored in previous examinations (40 to 99).
  3. `Extracurricular_Activities`: Participation in extracurricular activities (`Yes` / `No`).
  4. `Sleep_Hours`: Average sleep duration per day (4 to 9 hours).
  5. `Sample_Question_Papers_Practiced`: Number of practice mock papers solved (0 to 9).
- **Target Output:**
  - `Performance_Index`: Overall academic performance score (10.0 to 100.0).

---

## 📂 Project Directory Structure

```
d:/Project/student_performance_predictor/
├── data/
│   ├── download_dataset.py       # Downloads & caches the official Kaggle dataset
│   ├── Student_Performance.csv   # 10,000 real student records from Kaggle
│   └── sample_batch_test.csv     # Sample batch dataset (without target) for CSV upload demo
├── models/
│   ├── best_model.pkl            # Serialized scikit-learn Pipeline (Preprocessor + Regressor)
│   └── metrics.json              # Model comparison benchmarks (R², MAE, RMSE) & feature weights
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading, validation, and train-test splitting
│   └── train.py                  # Trains & benchmarks 5 ML models and saves best pipeline
├── app.py                        # Interactive Streamlit Web Application
├── requirements.txt              # Python dependencies list
└── README.md                     # Documentation & online deployment guide
```

---

## 🚀 Quickstart: Running Locally

### 1. Navigate to the Project Directory
```powershell
cd d:\Project\student_performance_predictor
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-run Dataset Download & Training
If you ever want to re-train the models from scratch:
```powershell
python data/download_dataset.py
python src/train.py
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will automatically launch in your browser at:
`http://localhost:8501`

---

## 🌐 Deploying Online for FREE (Streamlit Community Cloud)

You can host and share this machine learning project online so anyone can use it on the web:

### Step 1: Push Code to GitHub
1. Create a free account on [GitHub](https://github.com).
2. Create a new repository named `student-performance-predictor`.
3. Push your project files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Student Performance Predictor"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-predictor.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/student-performance-predictor`.
4. Set **Main file path** to: `app.py`.
5. Click **"Deploy!"**.

Streamlit will automatically build and host your web application with a live public URL (e.g. `https://student-performance-predictor.streamlit.app/`) that you can add to your portfolio or resume.

---

## 🏆 Model Benchmark Results

On the test set (1,975 real student records):

| Model | Test $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Best)** | **98.84%** | **1.65** | **2.08** |
| **Ridge Regression** | 98.84% | 1.65 | 2.08 |
| **Gradient Boosting** | 98.77% | 1.70 | 2.14 |
| **Random Forest** | 98.61% | 1.81 | 2.28 |
| **Decision Tree** | 97.29% | 2.55 | 3.17 |

---

## 💡 Key Machine Learning Insights

- **Primary Drivers:** Previous exam score ($\text{weight} \approx 17.62$) and hours studied ($\text{weight} \approx 7.37$) are the strongest predictors of student outcomes.
- **Sleep & Rest:** Consistent sleep (7–8 hours) prevents fatigue and supports high performance.
- **Practice Tests:** Practicing mock papers consistently yields a predictable score boost.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
