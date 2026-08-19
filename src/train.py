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
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Advanced Gradient Boosters
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from src.data_loader import (
    load_student_data,
    get_feature_target_split,
    split_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ALL_MODELS_SAVE_PATH = os.path.join(MODELS_DIR, "all_trained_models.pkl")
QUANTILE_MODELS_SAVE_PATH = os.path.join(MODELS_DIR, "quantile_models.pkl")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Standard Z-multipliers for confidence levels
Z_SCORES = {
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576
}

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
    Returns a comprehensive suite of regression models including Advanced Gradient Boosters.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting (GBDT)": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, random_state=42),
    }

    if XGB_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
            verbosity=0
        )

    if LGBM_AVAILABLE:
        models["LightGBM Regressor"] = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
            verbosity=-1
        )

    # Advanced Ensemble Stacking
    stack_estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)),
        ("gbdt", GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42))
    ]
    models["Stacking Ensemble Regressor"] = StackingRegressor(
        estimators=stack_estimators,
        final_estimator=Ridge(alpha=1.0)
    )

    return models

def train_quantile_regressors(X_train, y_train):
    """
    Trains non-parametric Quantile Gradient Boosters for uncertainty & interval bounds.
    """
    quantiles = [0.025, 0.05, 0.50, 0.95, 0.975]
    quantile_models = {}
    
    print("[*] Training Quantile Gradient Boosters for Prediction Intervals...")
    for q in quantiles:
        q_pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42
            ))
        ])
        q_pipe.fit(X_train, y_train)
        quantile_models[str(q)] = q_pipe
        print(f"      Fitted Quantile alpha={q:.3f}")
        
    return quantile_models

def compute_confidence_intervals(y_pred, residual_std, confidence_level=0.95):
    """
    Computes parametric Prediction Confidence Intervals with margin of error and clipping.
    """
    z = Z_SCORES.get(confidence_level, 1.960)
    margin = z * residual_std
    lower = np.clip(y_pred - margin, 10.0, 100.0)
    upper = np.clip(y_pred + margin, 10.0, 100.0)
    return lower, upper, margin

def train_and_evaluate():
    """
    End-to-end training, benchmarking, quantile modeling, and serialization pipeline.
    """
    print("==========================================================")
    print(" STUDENT PERFORMANCE PREDICTOR - CONFIDENCE INTERVALS & ML")
    print("==========================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("[1/5] Loading real Kaggle Student Performance dataset with Feature Engineering...")
    df = load_student_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    print(f"      Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    # 2. Train & Evaluate Candidates
    print("\n[2/5] Benchmarking baseline models & Advanced Gradient Boosters...")
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
        
        print(f"  --> {name:<28}: R² = {test_r2:.4f} | MAE = {test_mae:.4f} | RMSE = {test_rmse:.4f}")
    
    # 3. Select Best Model
    best_model_name = max(results, key=lambda k: results[k]["test_r2"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n[3/5] Champion Model Selected: '{best_model_name}' (Test R² = {best_metrics['test_r2']})")
    
    # 4. Train Quantile Regressors & Residual Diagnostics for Confidence Intervals
    print("\n[4/5] Training Quantile Regressors & Computing Confidence Interval Metrics...")
    quantile_models = train_quantile_regressors(X_train, y_train)
    
    # Residual analysis on test set
    y_test_pred = best_pipeline.predict(X_test)
    residuals = y_test.values - y_test_pred
    residual_std = float(np.std(residuals))
    residual_mean = float(np.mean(residuals))
    
    # Empirical Coverage Diagnostic
    coverage_diagnostics = {}
    for conf, z_val in Z_SCORES.items():
        margin = z_val * residual_std
        covered = np.mean((y_test.values >= (y_test_pred - margin)) & (y_test.values <= (y_test_pred + margin)))
        coverage_diagnostics[f"{int(conf*100)}%"] = {
            "target_coverage": conf,
            "empirical_coverage": round(float(covered), 4),
            "z_multiplier": z_val,
            "margin_of_error": round(float(margin), 2)
        }
        print(f"      Coverage at {int(conf*100)}% Nominal: {covered*100:.2f}% empirical (Margin ±{margin:.2f} pts)")

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
            
    # Sample Test Predictions with Intervals for Visual Plots (100 points)
    sample_test_indices = X_test.index[:100]
    sample_actual = y_test.loc[sample_test_indices].tolist()
    sample_preds_raw = best_pipeline.predict(X_test.loc[sample_test_indices])
    sample_predicted = [round(float(p), 2) for p in sample_preds_raw]
    
    # 95% Confidence Interval for sample
    sample_lower_95 = [round(float(p - 1.960 * residual_std), 2) for p in sample_preds_raw]
    sample_upper_95 = [round(float(p + 1.960 * residual_std), 2) for p in sample_preds_raw]
    
    # Save Metadata
    payload = {
        "best_model_name": best_model_name,
        "available_models": list(results.keys()),
        "all_model_benchmarks": results,
        "best_model_metrics": best_metrics,
        "feature_importances": feature_importance_dict,
        "confidence_intervals": {
            "residual_std": round(residual_std, 4),
            "residual_mean": round(residual_mean, 4),
            "coverage_diagnostics": coverage_diagnostics,
            "z_scores": Z_SCORES
        },
        "feature_names": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "sample_test_plot_data": {
            "actual": sample_actual,
            "predicted": sample_predicted,
            "lower_95": sample_lower_95,
            "upper_95": sample_upper_95
        },
        "dataset_info": {
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test)
        }
    }
    
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        
    print(f"\n[5/5] Saving champion pipeline to: {MODEL_SAVE_PATH}")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    
    print(f"      Saving all trained models bundle to: {ALL_MODELS_SAVE_PATH}")
    joblib.dump(fitted_pipelines, ALL_MODELS_SAVE_PATH)
    
    print(f"      Saving Quantile Regressors to: {QUANTILE_MODELS_SAVE_PATH}")
    joblib.dump(quantile_models, QUANTILE_MODELS_SAVE_PATH)
    
    print(f"      Saved benchmark & confidence interval metrics to: {METRICS_SAVE_PATH}")
    
    print("\n[OK] Model training & Prediction Confidence Interval calibration completed successfully!")
    return best_pipeline, payload

if __name__ == "__main__":
    train_and_evaluate()
