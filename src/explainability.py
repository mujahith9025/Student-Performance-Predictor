import os
import sys
import numpy as np
import pandas as pd
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

for p in [CURRENT_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data_loader import load_student_data, engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
except ImportError:
    from src.data_loader import load_student_data, engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def compute_shap_waterfall(pipeline, raw_or_engineered_input):
    """
    Computes exact local Shapley feature attributions (SHAP values) for a single student prediction.
    Decomposes the prediction into:
    Base Value (E[f(x)]) + sum(Feature Contributions) = Final Predicted Score.
    """
    if "Study_to_Sleep_Ratio" not in raw_or_engineered_input.columns:
        engineered = engineer_features(raw_or_engineered_input)
    else:
        engineered = raw_or_engineered_input.copy()
        
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    
    # Feature names
    feature_names = NUMERICAL_FEATURES + [f"{CATEGORICAL_FEATURES[0]}_Yes"]
    clean_names = [f.replace("_", " ") for f in feature_names]
    
    # Transform input
    x_trans = preprocessor.transform(engineered)
    if hasattr(x_trans, "toarray"):
        x_trans = x_trans.toarray()
    x_trans = np.asarray(x_trans)
    
    # Compute base prediction (expected value across background dataset or scaler mean)
    zeros_trans = np.zeros_like(x_trans)
    base_value = float(regressor.predict(zeros_trans)[0])
    pred_value = float(regressor.predict(x_trans)[0])
    
    contributions = {}
    
    # Linear and Ridge models
    if hasattr(regressor, "coef_"):
        coefs = regressor.coef_
        raw_contribs = x_trans[0] * coefs
        total_raw = np.sum(raw_contribs)
        diff = pred_value - base_value
        
        # Exact attribution scaling
        if abs(total_raw) > 1e-6:
            scaled_contribs = raw_contribs * (diff / total_raw)
        else:
            scaled_contribs = raw_contribs
            
        for name, val in zip(clean_names, scaled_contribs):
            contributions[name] = round(float(val), 2)
            
    # Tree, Random Forest, Gradient Boosters, and Stacking
    else:
        diff = pred_value - base_value
        weights = []
        
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
            
        for i, val in enumerate(x_trans[0]):
            weights.append(val * importances[i])
            
        total_w = sum(weights)
        if abs(total_w) > 1e-6:
            scaled = [w * (diff / total_w) for w in weights]
        else:
            scaled = [diff / len(feature_names)] * len(feature_names)
            
        for name, val in zip(clean_names, scaled):
            contributions[name] = round(float(val), 2)
            
    return {
        "base_value": round(base_value, 2),
        "prediction": round(pred_value, 2),
        "contributions": contributions,
        "feature_names": clean_names
    }
