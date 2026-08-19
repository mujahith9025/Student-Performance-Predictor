import os
import joblib
import pytest
from src.goal_solver import solve_target_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

def test_goal_solver_achievable_target():
    """Verify target score solver outputs 3 valid actionable pathways for achievable goals."""
    model = joblib.load(BEST_MODEL_PATH)
    res = solve_target_score(
        pipeline=model,
        previous_score=80,
        current_hours=3,
        current_sleep=7,
        current_papers=2,
        extracurricular="Yes",
        target_score=75.0
    )
    
    assert res["feasible"] is True
    assert res["target_score"] == 75.0
    assert len(res["pathways"]) == 3
    
    for p in res["pathways"]:
        assert p["predicted_score"] >= 74.0
        assert p["weekly_study_hours"] > 0
        assert p["required_hours"] >= 1
        assert p["required_papers"] >= 0

def test_goal_solver_impossible_target():
    """Verify solver handles boundary condition when target is unreachable."""
    model = joblib.load(BEST_MODEL_PATH)
    res = solve_target_score(
        pipeline=model,
        previous_score=40,
        current_hours=1,
        current_sleep=4,
        current_papers=0,
        extracurricular="No",
        target_score=99.0
    )
    
    assert "feasible" in res
    assert "current_pred" in res
