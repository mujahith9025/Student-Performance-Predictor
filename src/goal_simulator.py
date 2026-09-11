import numpy as np
import pandas as pd
from src.advanced_feature_engineering import engineer_features

def simulate_academic_goal(
    current_profile,
    target_score,
    preprocessor,
    model
):
    """
    Simulates multi-lever academic pathways to reach a target score using 14-feature profile:
    Adjusting Study Hours, Attendance, Test Prep, Tutoring, and Prerequisite Subject Scores.
    """
    # 1. Baseline Current Prediction
    df_current = pd.DataFrame([current_profile])
    df_current_eng = engineer_features(df_current)
    transformed_current = preprocessor.transform(df_current_eng)
    current_pred = float(np.clip(model.predict(transformed_current)[0], 0, 100))
    
    score_gap = round(target_score - current_pred, 1)
    
    # 2. Strategy A: Test Prep & Tutoring Support Boost
    test_prep_benefit = 0.0
    df_prep = df_current.copy()
    if current_profile.get("test preparation course", "none") == "none":
        df_prep["test preparation course"] = "completed"
        if current_profile.get("tutoring_support", "none") == "none":
            df_prep["tutoring_support"] = "peer_tutoring"
            
        df_prep_eng = engineer_features(df_prep)
        t_prep = preprocessor.transform(df_prep_eng)
        pred_with_prep = float(np.clip(model.predict(t_prep)[0], 0, 100))
        test_prep_benefit = round(pred_with_prep - current_pred, 1)
    else:
        pred_with_prep = current_pred
        
    # 3. Strategy B: Study Hours & Attendance Multi-Lever Optimization
    req_reading = float(current_profile.get("reading score", 65))
    req_writing = float(current_profile.get("writing score", 65))
    req_study_hours = float(current_profile.get("weekly_study_hours", 12.0))
    req_attendance = float(current_profile.get("attendance_rate", 85.0))
    
    if score_gap > 0:
        best_r = req_reading
        best_w = req_writing
        best_study = req_study_hours
        best_att = req_attendance
        
        for delta in range(0, 40):
            test_r = min(100.0, req_reading + delta)
            test_w = min(100.0, req_writing + delta)
            test_study = min(30.0, req_study_hours + (delta * 0.4))
            test_att = min(100.0, req_attendance + (delta * 0.3))
            
            df_test = df_prep.copy()
            df_test["reading score"] = test_r
            df_test["writing score"] = test_w
            df_test["weekly_study_hours"] = test_study
            df_test["attendance_rate"] = test_att
            df_test_eng = engineer_features(df_test)
            
            t_test = preprocessor.transform(df_test_eng)
            p_val = float(np.clip(model.predict(t_test)[0], 0, 100))
            
            if p_val >= target_score or (test_r == 100 and test_w == 100):
                best_r = test_r
                best_w = test_w
                best_study = test_study
                best_att = test_att
                break
                
        req_reading = best_r
        req_writing = best_w
        req_study_hours = round(best_study, 1)
        req_attendance = round(best_att, 1)
        
    # 4. Feasibility Classification
    if score_gap <= 0:
        feasibility = "Already Achieved"
        badge_color = "#059669"
        advice = "The student's current profile is already projected to meet or exceed this target!"
    elif score_gap <= 6:
        feasibility = "Easily Achievable (Minor Effort)"
        badge_color = "#10B981"
        advice = f"A minor boost of +{req_study_hours - current_profile.get('weekly_study_hours', 12.0):.1f} study hrs/week and completing the exam prep course will bridge this gap."
    elif score_gap <= 15:
        feasibility = "Moderate Effort Required"
        badge_color = "#3B82F6"
        advice = f"Achievable by boosting weekly study to {req_study_hours:.1f} hrs/week, maintaining {req_attendance:.0f}% attendance, and enrolling in tutoring support."
    else:
        feasibility = "Intensive Academic Intervention Needed"
        badge_color = "#DC2626"
        advice = f"Requires intensive structured tutoring, daily reading drills, dedicated {req_study_hours:.1f} study hrs/week, and full test preparation."
        
    return {
        "current_predicted_math": current_pred,
        "target_score": target_score,
        "score_gap": score_gap,
        "test_prep_benefit": test_prep_benefit,
        "required_reading_score": int(req_reading),
        "required_writing_score": int(req_writing),
        "required_study_hours": req_study_hours,
        "required_attendance": req_attendance,
        "reading_delta": int(req_reading - current_profile.get("reading score", 65)),
        "writing_delta": int(req_writing - current_profile.get("writing score", 65)),
        "study_delta": round(req_study_hours - current_profile.get("weekly_study_hours", 12.0), 1),
        "attendance_delta": round(req_attendance - current_profile.get("attendance_rate", 85.0), 1),
        "feasibility": feasibility,
        "badge_color": badge_color,
        "advice": advice
    }
