import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(df):
    """
    Creates rich domain-specific interaction features and non-linear synergy metrics.
    """
    df_feat = df.copy()
    
    # 1. Verbal Domain Aggregations & Synergies
    r = df_feat["reading score"]
    w = df_feat["writing score"]
    
    df_feat["verbal_average"] = (r + w) / 2.0
    df_feat["verbal_differential"] = (r - w)  # Divergence between reading and writing
    df_feat["verbal_synergy"] = np.sqrt(np.maximum(0, r * w))  # Geometric mean
    df_feat["verbal_ratio"] = r / (w + 1.0)
    
    # 2. Non-Linear Power Transformations (Curvature modeling)
    df_feat["reading_squared"] = (r / 100.0) ** 2 * 100.0
    df_feat["writing_squared"] = (w / 100.0) ** 2 * 100.0
    
    # 3. Ordinal Encoding for Parental Education
    edu_map = {
        "some high school": 1,
        "high school": 2,
        "some college": 3,
        "associate's degree": 4,
        "bachelor's degree": 5,
        "master's degree": 6
    }
    df_feat["parental_edu_rank"] = df_feat["parental level of education"].map(edu_map).fillna(3)
    
    # 4. Socioeconomic Readiness Composite Score
    prep_val = (df_feat["test preparation course"] == "completed").astype(float)
    lunch_val = (df_feat["lunch"] == "standard").astype(float)
    df_feat["socio_readiness_index"] = (lunch_val * 2.0) + (prep_val * 2.5) + (df_feat["parental_edu_rank"] * 0.8)
    
    # 5. Cross-Term Interaction Features
    df_feat["prep_x_reading"] = prep_val * r
    df_feat["lunch_x_writing"] = lunch_val * w
    
    return df_feat

def run_advanced_preprocessing():
    print("=" * 75)
    print("    STUDENT PERFORMANCE PREDICTOR - ADVANCED FEATURE ENGINEERING PIPELINE     ")
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
    
    print(f"[2] Engineered Features Generated: {X.shape[1]} features (from initial 7 raw features)")

    # 3. Identify Columns
    cat_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"    - Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"    - Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 4. Preprocessing Pipeline with Robust Scaling
    num_pipe = Pipeline([
        ("scaler", RobustScaler())  # Robust to outliers & extreme scores
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
    print(f"    - X_train shape: {X_train_trans.shape} (800 student samples, {X_train_trans.shape[1]} features)")
    print(f"    - X_test shape:  {X_test_trans.shape} (200 test samples)")

    # 6. Save Artifacts
    artifacts_dir = "artifacts"
    processed_dir = os.path.join("data", "processed")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_trans)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_trans)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy())

    print(f"\n[OK] Advanced Feature Engineering Completed! Transformed features: {X_train_trans.shape[1]}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_advanced_preprocessing()
