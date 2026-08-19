import pytest
from src.pdf_generator import create_student_pdf_report, create_goal_roadmap_pdf_report

def test_create_student_pdf_report():
    """Verify that student PDF report compiles into valid PDF binary."""
    pdf_bytes = create_student_pdf_report(
        student_name="Alex Test",
        subject="Mathematics & Statistics",
        model_name="Linear Regression",
        confidence_level="95%",
        pred_score=82.5,
        lower_bound=78.5,
        upper_bound=86.5,
        margin=4.0,
        letter_grade="A",
        standing_desc="Excellent",
        inputs={"Hours_Studied": 6, "Previous_Score": 80, "Sleep_Hours": 7, "Sample_Question_Papers_Practiced": 4, "Extracurricular_Activities": "Yes"},
        engineered_inputs={"Study_to_Sleep_Ratio": 0.86, "Practice_Density": 0.57, "Study_Effort_Score": 5.2},
        habit_balance_score=85.0,
        shap_contributions={"Hours Studied": 3.2, "Previous Score": 12.5},
        shap_base_value=54.8,
        recommendations=["Increase practice tests to +3 papers."],
        view_mode="Teacher & Counselor Mode",
        teacher_notes="Student shows high potential.",
        risk_flags=["None"]
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file does not have valid PDF magic bytes"

def test_create_goal_roadmap_pdf_report():
    """Verify that reverse goal roadmap PDF compiles cleanly."""
    pathways = [
        {
            "name": "Balanced Mastery Pathway",
            "tag": "Recommended",
            "description": "Optimal lifestyle routine.",
            "required_hours": 7.0,
            "required_papers": 5,
            "required_sleep": 7.5,
            "weekly_study_hours": 49.0,
            "predicted_score": 88.0,
            "delta_hours": 2.0,
            "delta_papers": 3
        }
    ]
    
    pdf_bytes = create_goal_roadmap_pdf_report(
        student_name="Alex Test",
        subject="General Academics",
        target_score=88.0,
        current_pred=72.0,
        gap=16.0,
        pathways=pathways
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
