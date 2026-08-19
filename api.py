import os
import sys
import json
import io
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Prepend project root and src directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path_entry in [SRC_DIR, PROJECT_ROOT]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from src.data_loader import (
    engineer_features,
    load_subject_data,
    SUBJECT_METADATA,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.explainability import compute_shap_waterfall
from src.goal_solver import solve_target_score
from src.pdf_generator import create_student_pdf_report, create_goal_roadmap_pdf_report
from src.db import (
    init_db,
    save_prediction_record,
    save_goal_record,
    fetch_prediction_history,
    fetch_goal_history,
    fetch_student_progress_timeline,
    get_database_stats
)

# Load Models & Benchmarks
SUBJECT_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "subject_models.pkl")
SUBJECT_METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "subject_metrics.json")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

# Z-scores for Confidence Intervals
Z_SCORES = {"80%": 1.282, "90%": 1.645, "95%": 1.960, "99%": 2.576}

app = FastAPI(
    title="🎓 Student Academic Performance Predictor — REST API Microservice",
    description=(
        "Production REST API for ML inference, Prediction Confidence Intervals, "
        "SHAP Explainability (XAI), Reverse Goal Solving, Database Persistence, "
        "and Dynamic Academic PDF Report Generation."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontend or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached runtime models
RUNTIME_SUBJECT_MODELS = {}
RUNTIME_SUBJECT_METRICS = {}

def get_runtime_models():
    global RUNTIME_SUBJECT_MODELS, RUNTIME_SUBJECT_METRICS
    if not RUNTIME_SUBJECT_MODELS:
        if os.path.exists(SUBJECT_MODELS_PATH) and os.path.exists(SUBJECT_METRICS_PATH):
            RUNTIME_SUBJECT_MODELS = joblib.load(SUBJECT_MODELS_PATH)
            with open(SUBJECT_METRICS_PATH, "r", encoding="utf-8") as f:
                RUNTIME_SUBJECT_METRICS = json.load(f)
        elif os.path.exists(BEST_MODEL_PATH) and os.path.exists(METRICS_PATH):
            best_m = joblib.load(BEST_MODEL_PATH)
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            RUNTIME_SUBJECT_MODELS = {"General Academics": {"Linear Regression": best_m}}
            RUNTIME_SUBJECT_METRICS = {"General Academics": m_data}
    return RUNTIME_SUBJECT_MODELS, RUNTIME_SUBJECT_METRICS

def compute_grade_info(score: float):
    if score >= 90: return "A+", "🌟 Outstanding Performance (Top Tier)"
    elif score >= 80: return "A", "🎯 Excellent (High Standing)"
    elif score >= 70: return "B", "👍 Good Performance (Above Average)"
    elif score >= 60: return "C", "⚖️ Satisfactory (Average)"
    elif score >= 50: return "D", "⚠️ Borderline / Needs Improvement"
    else: return "F", "🚨 At Risk (Action Required)"

# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------

class StudentInputSchema(BaseModel):
    student_name: str = Field("Alex Johnson", description="Student Name or Identifier")
    hours_studied: float = Field(..., ge=1.0, le=9.0, description="Daily study hours outside school (1 to 9)")
    previous_score: float = Field(..., ge=40.0, le=100.0, description="Previous assessment score (40 to 100)")
    sleep_hours: float = Field(..., ge=4.0, le=10.0, description="Daily sleep hours (4 to 10)")
    sample_papers: int = Field(..., ge=0, le=10, description="Number of practice mock papers completed (0 to 10)")
    extracurricular: str = Field("Yes", description="'Yes' or 'No'")
    subject: str = Field("General Academics", description="Academic Subject Discipline")
    model_name: Optional[str] = Field("Linear Regression", description="ML Algorithm to use for prediction")
    confidence_level: Optional[str] = Field("95%", description="Confidence Interval band: '80%', '90%', '95%', or '99%'")
    view_mode: Optional[str] = Field("Student Mode", description="'Student Mode' or 'Teacher & Counselor Mode'")
    teacher_notes: Optional[str] = Field("", description="Educator counseling remarks (optional)")
    save_to_db: Optional[bool] = Field(True, description="Whether to persist this prediction to SQLite / Supabase database")

    @field_validator("extracurricular")
    @classmethod
    def validate_extracurricular(cls, v):
        if str(v).strip().capitalize() not in ["Yes", "No"]:
            raise ValueError("extracurricular must be 'Yes' or 'No'")
        return str(v).strip().capitalize()

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, v):
        if str(v) not in Z_SCORES:
            raise ValueError(f"confidence_level must be one of {list(Z_SCORES.keys())}")
        return str(v)

class PredictionResponseSchema(BaseModel):
    success: bool
    student_name: str
    subject: str
    model_used: str
    confidence_level: str
    predicted_score: float
    lower_bound: float
    upper_bound: float
    margin_of_error: float
    letter_grade: str
    standing_desc: str
    habit_balance_score: float
    engineered_features: Dict[str, float]
    risk_flags: List[str]
    db_record_id: Optional[int] = None
    timestamp: str

class BatchPredictionSchema(BaseModel):
    records: List[StudentInputSchema]

class GoalSolverInputSchema(BaseModel):
    student_name: str = Field("Alex Johnson", description="Student Name or Identifier")
    previous_score: float = Field(..., ge=40.0, le=100.0, description="Current previous exam score (40-100)")
    current_hours: float = Field(4.0, ge=1.0, le=9.0, description="Current daily study hours")
    current_sleep: float = Field(7.0, ge=4.0, le=10.0, description="Current daily sleep hours")
    current_papers: int = Field(2, ge=0, le=10, description="Current practice mock papers")
    extracurricular: str = Field("Yes", description="'Yes' or 'No'")
    target_score: float = Field(..., ge=40.0, le=100.0, description="Desired Target Score to achieve")
    subject: Optional[str] = Field("General Academics", description="Academic Subject")

# -------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    """Root endpoint welcoming users and providing documentation links."""
    return {
        "title": "🎓 Student Academic Performance Predictor REST API",
        "status": "online",
        "version": "2.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "endpoints": [
            "POST /api/v1/predict",
            "POST /api/v1/predict/batch",
            "POST /api/v1/explain",
            "POST /api/v1/solve-goal",
            "GET  /api/v1/history",
            "GET  /api/v1/history/timeline/{student_name}",
            "GET  /api/v1/stats",
            "POST /api/v1/report/pdf",
            "GET  /api/v1/meta/subjects",
            "GET  /api/v1/meta/models"
        ]
    }

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for container orchestrators and load balancers."""
    models_bundle, _ = get_runtime_models()
    stats = get_database_stats()
    return {
        "status": "healthy",
        "models_loaded": len(models_bundle) > 0,
        "active_disciplines": list(models_bundle.keys()),
        "database_records": stats.get("total_predictions", 0),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/meta/subjects", tags=["Metadata"])
def get_subjects():
    """Returns all available academic subject disciplines and their descriptions."""
    return {"subjects": SUBJECT_METADATA}

@app.get("/api/v1/meta/models", tags=["Metadata"])
def get_models(subject: str = Query("General Academics", description="Academic Discipline")):
    """Returns available machine learning models and benchmark stats for the selected subject."""
    models_bundle, metrics_bundle = get_runtime_models()
    if subject not in models_bundle:
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' not found.")
    
    sub_metrics = metrics_bundle.get(subject, {})
    return {
        "subject": subject,
        "available_models": list(models_bundle[subject].keys()),
        "champion_model": sub_metrics.get("best_model_name", "Linear Regression"),
        "benchmarks": sub_metrics.get("all_model_benchmarks", {})
    }

@app.post("/api/v1/predict", response_model=PredictionResponseSchema, tags=["Inference"])
def predict_single_student(data: StudentInputSchema):
    """
    Predicts a student's score index, calculates calibrated confidence intervals,
    computes 6-D Habit Balance Score, detects Early Warning Risk Flags, and optionally
    persists the record to the SQLite database.
    """
    models_bundle, metrics_bundle = get_runtime_models()
    sub = data.subject if data.subject in models_bundle else "General Academics"
    sub_models = models_bundle.get(sub, models_bundle.get("General Academics", {}))
    
    model_name = data.model_name if data.model_name in sub_models else list(sub_models.keys())[0]
    model = sub_models[model_name]
    
    raw_df = pd.DataFrame([{
        "Hours_Studied": data.hours_studied,
        "Previous_Score": data.previous_score,
        "Sleep_Hours": data.sleep_hours,
        "Sample_Question_Papers_Practiced": data.sample_papers,
        "Extracurricular_Activities": data.extracurricular
    }])
    
    eng_df = engineer_features(raw_df)
    pred_score = float(model.predict(eng_df)[0])
    pred_score = max(10.0, min(100.0, round(pred_score, 1)))
    
    # Residual standard deviation for confidence interval
    sub_metrics = metrics_bundle.get(sub, {})
    residual_std = sub_metrics.get("confidence_intervals", {}).get("residual_std", 2.075)
    z = Z_SCORES.get(data.confidence_level, 1.960)
    margin = round(z * residual_std, 1)
    lower_bound = max(10.0, round(pred_score - margin, 1))
    upper_bound = min(100.0, round(pred_score + margin, 1))
    
    letter_grade, standing_desc = compute_grade_info(pred_score)
    
    # 6-D Habit Balance Score
    dim_study = min(100.0, (data.hours_studied / 9.0) * 100.0)
    dim_foundation = min(100.0, max(0.0, ((data.previous_score - 40.0) / 60.0) * 100.0))
    dim_practice = min(100.0, (data.sample_papers / 10.0) * 100.0)
    dim_sleep = min(100.0, max(0.0, 100.0 - abs(data.sleep_hours - 7.5) * 20.0))
    dim_effort = min(100.0, (eng_df["Study_Effort_Score"].iloc[0] / 9.4) * 100.0)
    dim_ec = 85.0 if data.extracurricular == "Yes" else 35.0
    habit_balance_score = round((dim_study + dim_foundation + dim_practice + dim_sleep + dim_effort + dim_ec) / 6.0, 1)
    
    # Early Warning Risk Flags
    risk_flags = []
    if pred_score < 50: risk_flags.append("🚨 High Academic Risk: Projected score is below passing threshold (50/100).")
    elif pred_score < 60: risk_flags.append("⚠️ Borderline Standing: At risk of underperforming without immediate intervention.")
    if data.sleep_hours < 6: risk_flags.append("😴 Severe Sleep Deficit: Sleeping less than 6 hours per day impairs cognitive retention.")
    if data.previous_score < 55: risk_flags.append("📝 Foundational Deficit: Low previous marks indicate need for fundamental concept remediation.")
    if data.hours_studied > 5 and data.sample_papers < 2: risk_flags.append("📄 Practice Imbalance: High reading hours but low active test practice.")
    
    db_id = None
    if data.save_to_db:
        try:
            db_id = save_prediction_record(
                student_name=data.student_name,
                subject=sub,
                model_used=model_name,
                hours_studied=data.hours_studied,
                previous_score=data.previous_score,
                sleep_hours=data.sleep_hours,
                sample_papers=data.sample_papers,
                extracurricular=data.extracurricular,
                predicted_score=pred_score,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=data.confidence_level,
                letter_grade=letter_grade,
                standing_desc=standing_desc,
                habit_balance_score=habit_balance_score,
                study_to_sleep_ratio=eng_df["Study_to_Sleep_Ratio"].iloc[0],
                practice_density=eng_df["Practice_Density"].iloc[0],
                study_effort_score=eng_df["Study_Effort_Score"].iloc[0],
                view_mode=data.view_mode,
                teacher_notes=data.teacher_notes,
                risk_flags=risk_flags
            )
        except Exception:
            db_id = None
            
    return PredictionResponseSchema(
        success=True,
        student_name=data.student_name,
        subject=sub,
        model_used=model_name,
        confidence_level=data.confidence_level,
        predicted_score=pred_score,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        margin_of_error=margin,
        letter_grade=letter_grade,
        standing_desc=standing_desc,
        habit_balance_score=habit_balance_score,
        engineered_features={
            "Study_to_Sleep_Ratio": round(float(eng_df["Study_to_Sleep_Ratio"].iloc[0]), 2),
            "Practice_Density": round(float(eng_df["Practice_Density"].iloc[0]), 2),
            "Study_Effort_Score": round(float(eng_df["Study_Effort_Score"].iloc[0]), 2)
        },
        risk_flags=risk_flags,
        db_record_id=db_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.post("/api/v1/predict/batch", tags=["Inference"])
def predict_batch(batch_data: BatchPredictionSchema):
    """Processes batch student predictions, calculating confidence intervals and risk prioritization."""
    results = []
    for item in batch_data.records:
        res = predict_single_student(item)
        results.append(res.model_dump())
        
    avg_score = round(float(np.mean([r["predicted_score"] for r in results])), 1) if results else 0.0
    pass_rate = round(float(np.mean([1 if r["predicted_score"] >= 50 else 0 for r in results]) * 100), 1) if results else 0.0
    
    return {
        "success": True,
        "total_records": len(results),
        "cohort_average_score": avg_score,
        "cohort_pass_rate": f"{pass_rate}%",
        "predictions": results
    }

@app.post("/api/v1/explain", tags=["Explainability (XAI)"])
def explain_prediction(data: StudentInputSchema):
    """
    Computes local SHAP (SHapley Additive exPlanations) waterfall feature attributions,
    showing exactly how each habit influenced the prediction relative to the cohort baseline.
    """
    models_bundle, _ = get_runtime_models()
    sub = data.subject if data.subject in models_bundle else "General Academics"
    model = models_bundle[sub].get(data.model_name, list(models_bundle[sub].values())[0])
    
    raw_df = pd.DataFrame([{
        "Hours_Studied": data.hours_studied,
        "Previous_Score": data.previous_score,
        "Sleep_Hours": data.sleep_hours,
        "Sample_Question_Papers_Practiced": data.sample_papers,
        "Extracurricular_Activities": data.extracurricular
    }])
    
    shap_data = compute_shap_waterfall(model, raw_df)
    eng_df = engineer_features(raw_df)
    pred_score = float(model.predict(eng_df)[0])
    
    contribs = shap_data.get("contributions", {})
    sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    top_pos = [k for k, v in sorted_c if v > 0][:2]
    top_neg = [k for k, v in sorted_c if v < 0][:2]
    
    return {
        "student_name": data.student_name,
        "subject": sub,
        "baseline_score": shap_data.get("base_value", 54.8),
        "predicted_score": round(pred_score, 1),
        "shap_contributions": contribs,
        "top_positive_drivers": top_pos,
        "top_negative_drags": top_neg
    }

@app.post("/api/v1/solve-goal", tags=["Goal Solver"])
def solve_goal_roadmap(data: GoalSolverInputSchema):
    """
    Solves the inverse academic optimization problem: Calculates the optimal
    combinations of daily study hours, mock papers, and sleep required to reach a target score.
    """
    models_bundle, _ = get_runtime_models()
    sub = data.subject if data.subject in models_bundle else "General Academics"
    model = list(models_bundle[sub].values())[0]
    
    goal_res = solve_target_score(
        pipeline=model,
        previous_score=data.previous_score,
        current_hours=data.current_hours,
        current_sleep=data.current_sleep,
        current_papers=data.current_papers,
        extracurricular=data.extracurricular,
        target_score=data.target_score
    )
    
    # Save goal to database
    if goal_res.get("pathways"):
        p0 = goal_res["pathways"][0]
        try:
            save_goal_record(
                student_name=data.student_name,
                subject=sub,
                current_pred=goal_res["current_pred"],
                target_score=data.target_score,
                score_gap=goal_res["gap"],
                recommended_pathway=p0["name"],
                required_hours=p0["required_hours"],
                required_papers=p0["required_papers"],
                required_sleep=p0["required_sleep"],
                weekly_hours=p0["weekly_study_hours"]
            )
        except Exception:
            pass
            
    return {
        "student_name": data.student_name,
        "subject": sub,
        "feasible": goal_res["feasible"],
        "current_projected_score": goal_res["current_pred"],
        "target_score": data.target_score,
        "score_gap": goal_res["gap"],
        "pathways": goal_res["pathways"],
        "message": goal_res.get("message", "Target solution computed successfully.")
    }

@app.get("/api/v1/history", tags=["Database & Records"])
def get_prediction_history(
    student_name: Optional[str] = Query(None, description="Filter by Student Name"),
    subject: Optional[str] = Query(None, description="Filter by Academic Subject"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return")
):
    """Queries persistent database prediction logs with optional name and subject filters."""
    df = fetch_prediction_history(student_name=student_name, subject=subject, limit=limit)
    return {
        "total_returned": len(df),
        "records": df.to_dict(orient="records")
    }

@app.get("/api/v1/history/timeline/{student_name}", tags=["Database & Records"])
def get_student_timeline(student_name: str):
    """Fetches chronological prediction progression for a single student for time-series charts."""
    df = fetch_student_progress_timeline(student_name=student_name)
    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No progress records found for student '{student_name}'.")
    return {
        "student_name": student_name,
        "total_assessments": len(df),
        "progression": df.to_dict(orient="records")
    }

@app.get("/api/v1/stats", tags=["Database & Records"])
def get_stats():
    """Returns high-level statistics about stored prediction audit records."""
    return get_database_stats()

@app.post("/api/v1/report/pdf", tags=["Reports"])
def generate_pdf_report(data: StudentInputSchema):
    """Generates an official academic diagnostic vector PDF report and returns the binary stream."""
    pred_res = predict_single_student(data)
    
    raw_dict = {
        "Hours_Studied": data.hours_studied,
        "Previous_Score": data.previous_score,
        "Sleep_Hours": data.sleep_hours,
        "Sample_Question_Papers_Practiced": data.sample_papers,
        "Extracurricular_Activities": data.extracurricular
    }
    
    pdf_bytes = create_student_pdf_report(
        student_name=pred_res.student_name,
        subject=pred_res.subject,
        model_name=pred_res.model_used,
        confidence_level=pred_res.confidence_level,
        pred_score=pred_res.predicted_score,
        lower_bound=pred_res.lower_bound,
        upper_bound=pred_res.upper_bound,
        margin=pred_res.margin_of_error,
        letter_grade=pred_res.letter_grade,
        standing_desc=pred_res.standing_desc,
        inputs=raw_dict,
        engineered_inputs=pred_res.engineered_features,
        habit_balance_score=pred_res.habit_balance_score,
        shap_contributions={},
        shap_base_value=54.8,
        recommendations=[f"Increase daily study hours to raise predicted score by +{round(data.hours_studied * 0.8, 1)} pts."],
        view_mode=data.view_mode,
        teacher_notes=data.teacher_notes,
        risk_flags=pred_res.risk_flags
    )
    
    filename = f"{pred_res.student_name.lower().replace(' ', '_')}_{pred_res.subject.lower().replace(' ', '_')}_report.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching Student Performance Predictor REST API on http://0.0.0.0:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
