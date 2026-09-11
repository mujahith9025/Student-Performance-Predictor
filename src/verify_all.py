import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.advanced_feature_engineering import engineer_features
from src.explainability import explain_single_student
from src.goal_simulator import simulate_academic_goal
from src.batch_predictor import process_batch_predictions, generate_sample_csv_template
from src.pdf_generator import generate_student_pdf_report

def verify_entire_system():
    print("=" * 70)
    print("     RUNNING SYSTEM-WIDE VERIFICATION SUITE (ALL ENGINES)     ")
    print("=" * 70)
    
    # 1. Load Artifacts
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    reg_model = joblib.load("artifacts/best_model.joblib")
    clf_model = joblib.load("artifacts/best_classifier.joblib")
    print("[1/6] Artifacts loaded successfully.")
    
    # 2. Test Single Student Inference
    sample_student = pd.DataFrame({
        "gender": ["female"],
        "race/ethnicity": ["group B"],
        "parental level of education": ["bachelor's degree"],
        "lunch": ["standard"],
        "test preparation course": ["completed"],
        "reading score": [82],
        "writing score": [85]
    })
    t_feat = preprocessor.transform(engineer_features(sample_student))
    score = reg_model.predict(t_feat)[0]
    prob = clf_model.predict_proba(t_feat)[0][1] * 100
    print(f"[2/6] Single Inference Verified: Score = {score:.2f}, Pass Prob = {prob:.1f}%")
    
    # 3. Test Explainability
    base_val, pred_val, contrib_df = explain_single_student(sample_student, preprocessor, reg_model)
    assert len(contrib_df) == 6, f"Expected 6 factors, got {len(contrib_df)}"
    print(f"[3/6] Explainable AI (XAI) Verified: {len(contrib_df)} local attributions computed.")
    
    # 4. Test Goal Simulator
    profile = sample_student.iloc[0].to_dict()
    goal_res = simulate_academic_goal(profile, 90, preprocessor, reg_model)
    assert "required_reading_score" in goal_res
    print(f"[4/6] Goal Simulator Verified: Target 90 requires Reading {goal_res['required_reading_score']}, Writing {goal_res['required_writing_score']}")
    
    # 5. Test Batch CSV Processor
    csv_str = generate_sample_csv_template()
    import io
    batch_in_df = pd.read_csv(io.StringIO(csv_str))
    proc_batch, summary = process_batch_predictions(batch_in_df, preprocessor, reg_model, clf_model)
    assert len(proc_batch) == 10
    print(f"[5/6] Batch Classroom Engine Verified: {summary['total_students']} students processed, Pass Rate = {summary['pass_rate']}%")
    
    # 6. Test PDF Generator
    pdf_bytes = generate_student_pdf_report(
        student_name="Alex Johnson",
        student_id="STU-2026-999",
        gender="female",
        race_ethnicity="group B",
        parental_education="bachelor's degree",
        lunch="standard",
        test_prep="completed",
        reading_score=82,
        writing_score=85,
        predicted_math=float(score),
        overall_avg=float((score + 82 + 85)/3.0),
        grade="A (Excellent)",
        model_name="Super-Stacking Meta-Regressor",
        tips=["Optimal standing across all subjects."],
        pass_prob=float(prob),
        risk_level="Safe / Low Risk"
    )
    assert len(pdf_bytes) > 1000, "PDF bytes too small"
    print(f"[6/6] Verified PDF Report Generated: {len(pdf_bytes)} bytes.")
    
    print("\n" + "=" * 70)
    print(" ALL 6 ENGINES PASSED 100% VERIFICATION WITH ZERO ERRORS! ")
    print("=" * 70)

if __name__ == "__main__":
    verify_entire_system()
