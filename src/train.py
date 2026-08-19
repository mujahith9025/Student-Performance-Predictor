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
    load_subject_data,
    create_subject_datasets_if_needed,
    get_feature_target_split,
    split_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    SUBJECT_METADATA
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ALL_MODELS_SAVE_PATH = os.path.join(MODELS_DIR, "all_trained_models.pkl")
QUANTILE_MODELS_SAVE_PATH = os.path.join(MODELS_DIR, "quantile_models.pkl")
SUBJECT_MODELS_SAVE_PATH = os.path.join(MODELS_DIR, "subject_models.pkl")
SUBJECT_METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "subject_metrics.json")
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
        
    return quantile_models

def train_subject_pipeline(subject_name: str, df: pd.DataFrame):
    """
    Trains and benchmarks models specifically for one academic subject.
    """
    print(f"\n---> Training Subject Pipeline: '{subject_name}' ({len(df)} records)...")
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    
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
        
    best_name = max(results, key=lambda k: results[k]["test_r2"])
    best_pipe = fitted_pipelines[best_name]
    
    # Residual analysis
    y_test_pred = best_pipe.predict(X_test)
    residuals = y_test.values - y_test_pred
    residual_std = float(np.std(residuals))
    residual_mean = float(np.mean(residuals))
    
    # Coverage Diagnostics
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

    # Feature weights
    feature_names = NUMERICAL_FEATURES + [f"{CATEGORICAL_FEATURES[0]}_Yes"]
    feature_importance_dict = {}
    reg = best_pipe.named_steps["regressor"]
    if hasattr(reg, "coef_"):
        for feat, coef in zip(feature_names, reg.coef_):
            feature_importance_dict[feat] = round(float(coef), 4)
    elif hasattr(reg, "feature_importances_"):
        for feat, imp in zip(feature_names, reg.feature_importances_):
            feature_importance_dict[feat] = round(float(imp), 4)
            
    # Sample Test predictions (100 rows)
    sample_idx = X_test.index[:100]
    sample_actual = y_test.loc[sample_idx].tolist()
    sample_preds_raw = best_pipe.predict(X_test.loc[sample_idx])
    sample_predicted = [round(float(p), 2) for p in sample_preds_raw]
    sample_lower_95 = [round(float(p - 1.960 * residual_std), 2) for p in sample_preds_raw]
    sample_upper_95 = [round(float(p + 1.960 * residual_std), 2) for p in sample_preds_raw]
    
    meta = SUBJECT_METADATA.get(subject_name, {})
    
    subject_summary = {
        "subject_name": subject_name,
        "icon": meta.get("icon", "📚"),
        "badge": meta.get("badge", "Academic"),
        "description": meta.get("description", ""),
        "focus": meta.get("focus", ""),
        "best_model_name": best_name,
        "best_model_metrics": results[best_name],
        "all_model_benchmarks": results,
        "feature_importances": feature_importance_dict,
        "confidence_intervals": {
            "residual_std": round(residual_std, 4),
            "residual_mean": round(residual_mean, 4),
            "coverage_diagnostics": coverage_diagnostics
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
    
    return fitted_pipelines, best_pipe, subject_summary

def train_and_evaluate_all():
    """
    Trains models for all academic subjects and saves global & subject artifacts.
    """
    print("==========================================================")
    print(" STUDENT PERFORMANCE PREDICTOR - MULTI-SUBJECT AI PIPELINE")
    print("==========================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    create_subject_datasets_if_needed()
    
    all_subject_models = {}
    all_subject_metrics = {}
    
    for subject_name in SUBJECT_METADATA.keys():
        df_sub = load_subject_data(subject_name)
        fitted_pipelines, best_pipe, sub_summary = train_subject_pipeline(subject_name, df_sub)
        all_subject_models[subject_name] = fitted_pipelines
        all_subject_metrics[subject_name] = sub_summary
        
    # Save Multi-Subject Artifacts
    print(f"\n[*] Saving multi-subject models bundle to: {SUBJECT_MODELS_SAVE_PATH}")
    joblib.dump(all_subject_models, SUBJECT_MODELS_SAVE_PATH)
    
    print(f"[*] Saving multi-subject metrics to: {SUBJECT_METRICS_SAVE_PATH}")
    with open(SUBJECT_METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_subject_metrics, f, indent=4)
        
    # Also save the default / General Academics artifacts for backward compatibility
    gen_summary = all_subject_metrics["General Academics"]
    gen_models = all_subject_models["General Academics"]
    best_gen_pipe = gen_models[gen_summary["best_model_name"]]
    
    joblib.dump(best_gen_pipe, MODEL_SAVE_PATH)
    joblib.dump(gen_models, ALL_MODELS_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(gen_summary, f, indent=4)
        
    # Quantile models for General Academics
    df_gen = load_subject_data("General Academics")
    X_gen, y_gen = get_feature_target_split(df_gen)
    X_tr, _, y_tr, _ = split_train_test(X_gen, y_gen, test_size=0.2, random_state=42)
    quantile_models = train_quantile_regressors(X_tr, y_tr)
    joblib.dump(quantile_models, QUANTILE_MODELS_SAVE_PATH)
    
    print("\n[OK] All Multi-Subject ML pipelines & artifacts trained and saved successfully!")

if __name__ == "__main__":
    train_and_evaluate_all()
