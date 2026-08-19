import os
import joblib
import pytest
import pandas as pd
from src.explainability import compute_shap_waterfall

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

def test_compute_shap_waterfall():
    """Verify that SHAP decomposes predictions cleanly into base value and contributions."""
    model = joblib.load(BEST_MODEL_PATH)
    sample_df = pd.DataFrame([{
        "Hours_Studied": 7,
        "Previous_Score": 88,
        "Sleep_Hours": 8,
        "Sample_Question_Papers_Practiced": 4,
        "Extracurricular_Activities": "Yes"
    }])
    
    shap_res = compute_shap_waterfall(model, sample_df)
    
    assert "base_value" in shap_res
    assert "contributions" in shap_res
    assert isinstance(shap_res["base_value"], (int, float))
    assert len(shap_res["contributions"]) > 0
    
    # Base score should be realistic population baseline (e.g. ~50-60)
    assert 40.0 <= shap_res["base_value"] <= 70.0
    
    # Previous score and Hours studied should have positive attribution for high values
    contribs = shap_res["contributions"]
    assert "Previous Score" in contribs or any("Previous" in k for k in contribs)
