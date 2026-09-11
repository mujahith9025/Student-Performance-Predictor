import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_CATEGORICAL_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "internet_access",
    "extracurricular_activities",
    "tutoring_support"
]

RAW_NUMERICAL_COLS = [
    "reading score",
    "writing score",
    "attendance_rate",
    "weekly_study_hours",
    "sleep_hours_per_day",
    "past_failures"
]

def engineer_features(df):
    """
    Creates rich domain-specific interaction features, behavioral engagement scores,
    and non-linear synergy metrics across 14 input dimensions.
    """
    df_feat = df.copy()
    
    # 1. Fill missing columns with realistic defaults if legacy CSV is provided
    defaults = {
        "gender": "female",
        "race/ethnicity": "group C",
        "parental level of education": "some college",
        "lunch": "standard",
        "test preparation course": "none",
        "internet_access": "yes",
        "extracurricular_activities": "no",
        "tutoring_support": "none",
        "reading score": 65.0,
        "writing score": 65.0,
        "attendance_rate": 85.0,
        "weekly_study_hours": 12.0,
        "sleep_hours_per_day": 7.5,
        "past_failures": 0
    }
    for col, default_val in defaults.items():
        if col not in df_feat.columns:
            df_feat[col] = default_val
            
    r = df_feat["reading score"].astype(float)
    w = df_feat["writing score"].astype(float)
    att = df_feat["attendance_rate"].astype(float)
    study = df_feat["weekly_study_hours"].astype(float)
    sleep = df_feat["sleep_hours_per_day"].astype(float)
    fails = df_feat["past_failures"].astype(float)
    
    # 2. Verbal Domain Synergies & Ratios
    df_feat["verbal_average"] = (r + w) / 2.0
    df_feat["verbal_differential"] = (r - w)
    df_feat["verbal_synergy"] = np.sqrt(np.maximum(0, r * w))
    df_feat["verbal_ratio"] = r / (w + 1.0)
    
    # 3. Non-Linear Curvature Transformations
    df_feat["reading_squared"] = (r / 100.0) ** 2 * 100.0
    df_feat["writing_squared"] = (w / 100.0) ** 2 * 100.0
    
    # 4. Ordinal Parent Education Rank
    edu_map = {
        "some high school": 1,
        "high school": 2,
        "some college": 3,
        "associate's degree": 4,
        "bachelor's degree": 5,
        "master's degree": 6
    }
    df_feat["parental_edu_rank"] = df_feat["parental level of education"].map(edu_map).fillna(3).astype(float)
    
    # 5. Engagement & Academic Effort Metrics
    prep_val = (df_feat["test preparation course"] == "completed").astype(float)
    lunch_val = (df_feat["lunch"] == "standard").astype(float)
    net_val = (df_feat["internet_access"] == "yes").astype(float)
    extra_val = (df_feat["extracurricular_activities"] == "yes").astype(float)
    
    tutor_weight = {
        "none": 0.0,
        "peer_tutoring": 1.5,
        "private_tutor": 3.0
    }
    tutor_val = df_feat["tutoring_support"].map(tutor_weight).fillna(0.0).astype(float)
    
    # Study & Attendance Synergy
    df_feat["study_attendance_synergy"] = (study / 20.0) * (att / 100.0) * 10.0
    df_feat["academic_effort_index"] = (study * 0.5) + (att * 0.1) + (prep_val * 3.0) + (tutor_val * 2.0)
    
    # 6. Behavioral Risk & Friction Metric
    df_feat["academic_risk_friction"] = (fails * 4.0) + np.maximum(0.0, 75.0 - att) * 0.25
    
    # 7. Wellness & Lifestyle Balance
    # Optimal sleep between 7.0 and 8.5 hours
    sleep_balance = np.where(
        sleep < 6.0, (sleep - 6.0) * 2.0,
        np.where(sleep > 9.0, -1.0, 1.5)
    )
    df_feat["wellness_lifestyle_score"] = sleep_balance + (lunch_val * 2.0) + (extra_val * 1.0)
    
    # 8. Socioeconomic Readiness Composite
    df_feat["socio_readiness_index"] = (lunch_val * 2.0) + (net_val * 1.5) + (df_feat["parental_edu_rank"] * 0.8)
    
    # 9. Cross-Interaction Terms
    df_feat["prep_x_reading"] = prep_val * r
    df_feat["lunch_x_writing"] = lunch_val * w
    df_feat["study_x_attendance"] = study * (att / 100.0)
    df_feat["tutor_x_study"] = tutor_val * study
    
    return df_feat

def run_advanced_preprocessing():
    print("=" * 75)
    print("    STUDENT PERFORMANCE PREDICTOR - ADVANCED 14-FEATURE PIPELINE     ")
    print("=" * 75)
    sys.stdout.flush()

    # 1. Load Raw Data
    raw_path = os.path.join("data", "StudentsPerformance.csv")
    df = pd.read_csv(raw_path)
    print(f"\n[1] Loaded Raw Data: {df.shape[0]} rows, {df.shape[1]} raw columns")

    # 2. Apply Domain Feature Engineering
    df_engineered = engineer_features(df)
    target_column = "math score"
    
    X = df_engineered.drop(columns=[target_column])
    y = df_engineered[target_column]
    
    print(f"[2] Engineered Features Generated: {X.shape[1]} features (from 14 raw features)")

    # 3. Identify Columns
    cat_cols = RAW_CATEGORICAL_COLS
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"    - Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"    - Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 4. Preprocessing Pipeline with Robust Scaling
    num_pipe = Pipeline([
        ("scaler", RobustScaler())
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # 5. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    print(f"\n[3] Final Preprocessed Feature Matrix:")
    print(f"    - X_train: {X_train_trans.shape} (1600 samples)")
    print(f"    - X_test:  {X_test_trans.shape} (400 samples)")

    # Save artifacts
    artifacts_dir = "artifacts"
    processed_dir = os.path.join("data", "processed")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_trans)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_trans)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy())
    
    print("\n[OK] Advanced 14-Feature Preprocessing Completed Successfully!")
    sys.stdout.flush()

if __name__ == "__main__":
    run_advanced_preprocessing()
