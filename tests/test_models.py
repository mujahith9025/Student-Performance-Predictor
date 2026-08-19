import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np
from src.data_loader import engineer_features, SUBJECT_METADATA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_trained_models.pkl")
SUBJECT_MODELS_PATH = os.path.join(MODELS_DIR, "subject_models.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
SUBJECT_METRICS_PATH = os.path.join(MODELS_DIR, "subject_metrics.json")

def test_model_files_exist():
    """Verify that all required serialized model artifacts exist."""
    assert os.path.exists(BEST_MODEL_PATH), "best_model.pkl is missing"
    assert os.path.exists(ALL_MODELS_PATH), "all_trained_models.pkl is missing"
    assert os.path.exists(SUBJECT_MODELS_PATH), "subject_models.pkl is missing"
    assert os.path.exists(METRICS_PATH), "metrics.json is missing"
    assert os.path.exists(SUBJECT_METRICS_PATH), "subject_metrics.json is missing"

def test_best_model_prediction():
    """Test inference with champion model on single student input."""
    model = joblib.load(BEST_MODEL_PATH)
    sample_raw = pd.DataFrame([{
        "Hours_Studied": 5,
        "Previous_Score": 75,
        "Sleep_Hours": 7,
        "Sample_Question_Papers_Practiced": 3,
        "Extracurricular_Activities": "Yes"
    }])
    sample_eng = engineer_features(sample_raw)
    pred = float(model.predict(sample_eng)[0])
    
    assert 10.0 <= pred <= 100.0, f"Predicted score {pred} outside valid range [10, 100]"
    assert 50.0 <= pred <= 85.0, f"Expected realistic score, got {pred}"

def test_all_trained_models_bundle():
    """Verify that all 9 models in all_trained_models.pkl can execute predictions."""
    models_dict = joblib.load(ALL_MODELS_PATH)
    assert len(models_dict) >= 9
    
    sample_eng = engineer_features(pd.DataFrame([{
        "Hours_Studied": 6,
        "Previous_Score": 85,
        "Sleep_Hours": 8,
        "Sample_Question_Papers_Practiced": 5,
        "Extracurricular_Activities": "Yes"
    }]))
    
    for name, m in models_dict.items():
        pred = float(m.predict(sample_eng)[0])
        assert 10.0 <= pred <= 100.0, f"Model {name} produced out of bound prediction: {pred}"

def test_subject_models_bundle():
    """Verify that each academic subject has trained models and valid predictions."""
    subject_bundle = joblib.load(SUBJECT_MODELS_PATH)
    for sub in SUBJECT_METADATA.keys():
        assert sub in subject_bundle, f"Subject {sub} not found in bundle"
        sub_models = subject_bundle[sub]
        assert len(sub_models) >= 1
        
        sample_eng = engineer_features(pd.DataFrame([{
            "Hours_Studied": 4,
            "Previous_Score": 60,
            "Sleep_Hours": 6,
            "Sample_Question_Papers_Practiced": 2,
            "Extracurricular_Activities": "No"
        }]))
        
        for m_name, model in sub_models.items():
            pred = float(model.predict(sample_eng)[0])
            assert 10.0 <= pred <= 100.0

def test_metrics_json_integrity():
    """Verify test metrics R2, MAE, and RMSE benchmarks meet high quality standards."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    assert "best_model_name" in metrics
    best_m = metrics.get("best_model_metrics", {})
    assert best_m.get("test_r2", 0) > 0.95, f"R2 score too low: {best_m.get('test_r2')}"
    assert best_m.get("test_mae", 99) < 2.5, f"MAE error too high: {best_m.get('test_mae')}"
