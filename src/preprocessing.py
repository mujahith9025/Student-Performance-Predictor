import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data():
    print("=" * 65)
    print("      STUDENT PERFORMANCE PREDICTOR - DATA PREPROCESSING      ")
    print("=" * 65)
    sys.stdout.flush()
    
    # 1. Load the raw dataset
    raw_data_path = os.path.join("data", "StudentsPerformance.csv")
    df = pd.read_csv(raw_data_path)
    print(f"\n[1] Loaded raw data: {raw_data_path} (Shape: {df.shape})")
    sys.stdout.flush()

    # 2. Separate Features (X) and Target (y)
    target_column = "math score"
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\n[2] Features (X) count: {X.shape[1]} columns")
    print(f"    Target (y) column: '{target_column}'")

    # 3. Identify Numerical and Categorical Columns
    num_features = ["reading score", "writing score"]
    cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
    
    print(f"\n    - Numerical Features ({len(num_features)}): {num_features}")
    print(f"    - Categorical Features ({len(cat_features)}): {cat_features}")
    sys.stdout.flush()

    # 4. Define Preprocessing Pipelines
    # Numerical: Standardize (Mean = 0, Std = 1)
    # Categorical: Convert discrete categories to One-Hot encoded binary features
    num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("one_hot_encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, num_features),
            ("cat_pipeline", cat_pipeline, cat_features)
        ]
    )

    # 5. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[3] Train-Test Split (80/20):")
    print(f"    - X_train shape: {X_train.shape} (800 samples)")
    print(f"    - X_test shape:  {X_test.shape} (200 samples)")
    print(f"    - y_train shape: {y_train.shape}")
    print(f"    - y_test shape:  {y_test.shape}")
    sys.stdout.flush()

    # 6. Fit Preprocessor on Training Data & Transform both sets
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract encoded feature names for explainability
    cat_encoder = preprocessor.named_transformers_['cat_pipeline'].named_steps['one_hot_encoder']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names
    
    print(f"\n[4] Transformation Results:")
    print(f"    - Transformed feature count: {X_train_transformed.shape[1]} features (from original {X.shape[1]})")
    print("    - Encoded Feature List:")
    for i, col_name in enumerate(all_feature_names, 1):
        print(f"       {i:02d}. {col_name}")
    sys.stdout.flush()

    # 7. Save Processed Datasets & Preprocessor Pipeline
    artifacts_dir = "artifacts"
    processed_dir = os.path.join("data", "processed")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    preprocessor_file = os.path.join(artifacts_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_file)
    print(f"\n[5] Saved Preprocessor Pipeline to: {preprocessor_file}")

    # Save processed arrays for Phase 4 (Model Building)
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_transformed)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_transformed)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy())
    
    # Also save raw train & test splits as CSV for reference
    X_train.to_csv(os.path.join(processed_dir, "train_features.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "test_features.csv"), index=False)
    
    print(f"    Saved processed matrices to: {processed_dir}")
    print("\n[OK] Phase 3 Preprocessing Successfully Completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    preprocess_data()
