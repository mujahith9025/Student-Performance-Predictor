import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data_loader import (
    load_student_data,
    get_feature_target_split,
    split_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "metrics.json")

def build_preprocessor():
    """
    Constructs a ColumnTransformer for preprocessing numerical and categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def get_candidate_models():
    """
    Returns a dictionary of candidate regression models to benchmark.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

def train_and_evaluate():
    """
    End-to-end training, benchmarking, and serialization pipeline.
    """
    print("==================================================")
    print("   STUDENT PERFORMANCE PREDICTOR - MODEL TRAINING ")
    print("==================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("[1/4] Loading real Kaggle Student Performance dataset...")
    df = load_student_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    print(f"      Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    # 2. Train & Evaluate Candidates
    print("\n[2/4] Training and benchmarking candidate models...")
    candidate_models = get_candidate_models()
    results = {}
    fitted_pipelines = {}
    
    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model)
        ])
        
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        
        results[name] = {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_mse": round(test_mse, 4)
        }
        
        print(f"  --> {name:<20}: R² = {test_r2:.4f} | MAE = {test_mae:.4f} | RMSE = {test_rmse:.4f}")
    
    # 3. Select Best Model
    # On this dataset, Linear Regression and Ridge achieve R² ~ 0.9888 with lowest MAE
    best_model_name = max(results, key=lambda k: results[k]["test_r2"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n[3/4] Best Model Selected: '{best_model_name}' (Test R² = {best_metrics['test_r2']})")
    
    # Extract Feature Importances / Coefficients
    feature_names = NUMERICAL_FEATURES + [f"{CATEGORICAL_FEATURES[0]}_Yes"]
    feature_importance_dict = {}
    
    regressor = best_pipeline.named_steps["regressor"]
    if hasattr(regressor, "coef_"):
        coefficients = regressor.coef_
        for feat, coef in zip(feature_names, coefficients):
            feature_importance_dict[feat] = round(float(coef), 4)
    elif hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        for feat, imp in zip(feature_names, importances):
            feature_importance_dict[feat] = round(float(imp), 4)
            
    # Sample Test Predictions for UI Plots (100 points)
    sample_test_indices = X_test.index[:100]
    sample_actual = y_test.loc[sample_test_indices].tolist()
    sample_predicted = [round(float(p), 2) for p in best_pipeline.predict(X_test.loc[sample_test_indices])]
    
    # Save Metadata
    payload = {
        "best_model_name": best_model_name,
        "all_model_benchmarks": results,
        "best_model_metrics": best_metrics,
        "feature_importances": feature_importance_dict,
        "feature_names": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "sample_test_plot_data": {
            "actual": sample_actual,
            "predicted": sample_predicted
        },
        "dataset_info": {
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test)
        }
    }
    
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    print(f"[4/4] Saving best pipeline to: {MODEL_SAVE_PATH}")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"      Saved metrics summary to: {METRICS_SAVE_PATH}")
    
    print("\n[OK] Training completed successfully!")
    return best_pipeline, payload

if __name__ == "__main__":
    train_and_evaluate()
