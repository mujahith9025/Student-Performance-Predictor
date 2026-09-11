import os
import sys
import io
import pandas as pd
import numpy as np
import joblib
from src.advanced_feature_engineering import engineer_features
from src.explainability import explain_single_student
from src.goal_simulator import simulate_academic_goal
from src.batch_predictor import process_batch_predictions, generate_sample_csv_template, load_or_create_sample_cohort
from src.sample_generator import generate_synthetic_classroom
from src.prescriptive_solutions import (
    diagnose_student_weaknesses,
    generate_prescriptive_solution,
    generate_classroom_intervention_matrix
)
from src.pdf_generator import generate_student_pdf_report, generate_classroom_pdf_report

def verify_entire_system():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print("=" * 75)
    print("     RUNNING SYSTEM-WIDE VERIFICATION SUITE (ALL 10 ENGINES)     ")
    print("=" * 75)
    
    # 1. Load Artifacts
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    reg_model = joblib.load("artifacts/best_model.joblib")
    clf_model = joblib.load("artifacts/best_classifier.joblib")
    print("[1/8] Artifacts loaded successfully.")
    
    # 2. Test Single Student Inference with 14 Features
    sample_student = pd.DataFrame({
        "gender": ["female"],
        "race/ethnicity": ["group B"],
        "parental level of education": ["bachelor's degree"],
        "lunch": ["standard"],
        "test preparation course": ["completed"],
        "internet_access": ["yes"],
        "extracurricular_activities": ["yes"],
        "tutoring_support": ["peer_tutoring"],
        "attendance_rate": [94.0],
        "weekly_study_hours": [18.0],
        "sleep_hours_per_day": [7.8],
        "past_failures": [0],
        "reading score": [82],
        "writing score": [85]
    })
    t_feat = preprocessor.transform(engineer_features(sample_student))
    score = float(reg_model.predict(t_feat)[0])
    prob = float(clf_model.predict_proba(t_feat)[0][1] * 100)
    print(f"[2/8] Single Inference Verified: Score = {score:.2f}, Pass Prob = {prob:.1f}%")
    
    # 3. Test Explainability (XAI)
    base_val, pred_val, contrib_df = explain_single_student(sample_student, preprocessor, reg_model)
    assert len(contrib_df) >= 8, f"Expected >=8 factors, got {len(contrib_df)}"
    print(f"[3/8] Explainable AI (XAI) Verified: {len(contrib_df)} local attributions computed.")
    
    # 4. Test Goal Simulator
    profile = sample_student.iloc[0].to_dict()
    goal_res = simulate_academic_goal(profile, 90, preprocessor, reg_model)
    assert "required_reading_score" in goal_res
    print(f"[4/8] Goal Simulator Verified: Target 90 requires Reading {goal_res['required_reading_score']}, Study {goal_res['required_study_hours']}h/wk")
    
    # 5. Test AI Prescriptive Solutions Engine
    sol_res = generate_prescriptive_solution(profile, predicted_math=score, pass_prob=prob, grade="A (Excellent)")
    assert "study_hours" in sol_res
    assert "interventions" in sol_res
    assert "projected_score" in sol_res
    print(f"[5/8] Prescriptive Solutions Verified: Projected Score = {sol_res['projected_score']:.1f}, Weekly Hours = {sol_res['total_study_hours']:.1f}h")
    
    # 6. Test Synthetic Classroom Generator (14 Features)
    syn_df = generate_synthetic_classroom(n_students=50, cohort_type="balanced", seed=42)
    assert len(syn_df) == 50
    assert "student_name" in syn_df.columns
    assert "attendance_rate" in syn_df.columns
    print(f"[6/8] Dynamic Synthetic Cohort Generator Verified: {len(syn_df)} students with 14 features generated.")
    
    # 7. Test Batch Classroom & Intervention Matrix
    proc_batch, summary = process_batch_predictions(syn_df, preprocessor, reg_model, clf_model)
    matrix = generate_classroom_intervention_matrix(proc_batch)
    assert len(proc_batch) == 50
    assert "Prescribed_Intervention" in proc_batch.columns
    print(f"[7/8] Batch Analytics & Intervention Matrix Verified: {summary['total_students']} students processed, Pass Rate = {summary['pass_rate']}%, At-Risk = {summary['at_risk_count']}")
    
    # 8. Test Enhanced PDF Generator (Single & Classroom)
    pdf_single = generate_student_pdf_report(
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
        risk_level="Safe / Low Risk",
        prescriptive_solution=sol_res,
        attendance_rate=94.0,
        weekly_study_hours=18.0,
        sleep_hours_per_day=7.8,
        past_failures=0,
        tutoring_support="peer_tutoring",
        internet_access="yes"
    )
    assert len(pdf_single) > 1000
    
    pdf_class = generate_classroom_pdf_report(
        classroom_df=proc_batch,
        summary=summary,
        cohort_name="Verification Synthetic Cohort",
        intervention_matrix=matrix
    )
    assert len(pdf_class) > 1000
    print(f"[8/10] Verified PDF Suite: Single Cert = {len(pdf_single)} bytes, Class Report = {len(pdf_class)} bytes.")
    
    # 9. Test Conformal Prediction Uncertainty Quantifier
    from src.advanced_ml_statistical import predict_with_confidence_intervals, classify_student_archetype
    unc_dict = joblib.load("artifacts/uncertainty_model.joblib")
    ci_res = predict_with_confidence_intervals(t_feat, reg_model, unc_dict)
    assert "ci_95_range" in ci_res
    print(f"[9/10] Statistical Uncertainty Engine Verified: 95% CI = {ci_res['ci_95_str']}")
    
    # 10. Test Unsupervised Behavioral Archetype Clusterer
    cluster_bundle = joblib.load("artifacts/archetype_clusterer.joblib")
    arch_res = classify_student_archetype(t_feat, cluster_bundle)
    assert "name" in arch_res
    assert "affinity_score" in arch_res
    print(f"[10/10] Behavioral Archetype Engine Verified: Assigned to '{arch_res['name']}' (Affinity: {arch_res['affinity_score']:.1f}%)")
    
    print("\n" + "=" * 75)
    print(" ALL 10 ENGINES PASSED 100% VERIFICATION WITH ZERO ERRORS! ")
    print("=" * 75)

if __name__ == "__main__":
    verify_entire_system()

