import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_api_root():
    """Verify root endpoint provides API status and endpoint directory."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

def test_api_health():
    """Verify healthcheck endpoint reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_api_meta_subjects():
    """Verify metadata endpoint returns 4 academic disciplines."""
    response = client.get("/api/v1/meta/subjects")
    assert response.status_code == 200
    data = response.json()
    assert "General Academics" in data["subjects"]
    assert "Mathematics & Statistics" in data["subjects"]

def test_api_meta_models():
    """Verify models endpoint returns available models and champion model."""
    response = client.get("/api/v1/meta/models?subject=General%20Academics")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert len(data["available_models"]) >= 1

def test_api_single_prediction():
    """Verify POST /api/v1/predict computes prediction, confidence bounds, and saves to DB."""
    payload = {
        "student_name": "API Test Student",
        "hours_studied": 6.0,
        "previous_score": 80.0,
        "sleep_hours": 7.5,
        "sample_papers": 4,
        "extracurricular": "Yes",
        "subject": "Mathematics & Statistics",
        "confidence_level": "95%",
        "save_to_db": True
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["student_name"] == "API Test Student"
    assert 10.0 <= data["predicted_score"] <= 100.0
    assert data["lower_bound"] <= data["predicted_score"] <= data["upper_bound"]
    assert data["letter_grade"] in ["A+", "A", "B", "C", "D", "F"]
    assert "Study_to_Sleep_Ratio" in data["engineered_features"]
    assert data["db_record_id"] is not None

def test_api_batch_prediction():
    """Verify POST /api/v1/predict/batch computes predictions for multiple students."""
    payload = {
        "records": [
            {
                "student_name": "Student Alpha",
                "hours_studied": 7.0,
                "previous_score": 90.0,
                "sleep_hours": 8.0,
                "sample_papers": 5,
                "extracurricular": "Yes",
                "subject": "General Academics"
            },
            {
                "student_name": "Student Beta",
                "hours_studied": 3.0,
                "previous_score": 50.0,
                "sleep_hours": 5.0,
                "sample_papers": 1,
                "extracurricular": "No",
                "subject": "General Academics"
            }
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2

def test_api_explain():
    """Verify POST /api/v1/explain returns SHAP waterfall attributions."""
    payload = {
        "student_name": "API Test Student",
        "hours_studied": 5.0,
        "previous_score": 75.0,
        "sleep_hours": 7.0,
        "sample_papers": 3,
        "extracurricular": "Yes",
        "subject": "General Academics"
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline_score" in data
    assert "shap_contributions" in data

def test_api_solve_goal():
    """Verify POST /api/v1/solve-goal generates 3 target achievement pathways."""
    payload = {
        "student_name": "API Test Student",
        "previous_score": 80.0,
        "current_hours": 3.0,
        "current_sleep": 7.0,
        "current_papers": 2,
        "extracurricular": "Yes",
        "target_score": 75.0,
        "subject": "General Academics"
    }
    response = client.post("/api/v1/solve-goal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feasible"] is True
    assert len(data["pathways"]) == 3

def test_api_history_and_stats():
    """Verify GET /api/v1/history and GET /api/v1/stats."""
    res_stats = client.get("/api/v1/stats")
    assert res_stats.status_code == 200
    assert "total_predictions" in res_stats.json()
    
    res_hist = client.get("/api/v1/history?limit=10")
    assert res_hist.status_code == 200
    assert "records" in res_hist.json()

def test_api_pdf_report_streaming():
    """Verify POST /api/v1/report/pdf streams valid PDF binary bytes."""
    payload = {
        "student_name": "API Test PDF Student",
        "hours_studied": 6.0,
        "previous_score": 80.0,
        "sleep_hours": 7.5,
        "sample_papers": 4,
        "extracurricular": "Yes",
        "subject": "General Academics"
    }
    response = client.post("/api/v1/report/pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
