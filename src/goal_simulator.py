import numpy as np
import pandas as pd
import joblib
from src.advanced_feature_engineering import engineer_features

def simulate_academic_goal(
    current_profile,
    target_score,
    preprocessor,
    model
):
    """
    Simulates actionable academic pathways to reach a target score using advanced features.
    """
    # 1. Baseline Current Prediction
    df_current = pd.DataFrame([current_profile])
    df_current_eng = engineer_features(df_current)
    transformed_current = preprocessor.transform(df_current_eng)
    current_pred = float(np.clip(model.predict(transformed_current)[0], 0, 100))
    
    score_gap = round(target_score - current_pred, 1)
    
    # 2. Strategy A: What if the student completes Test Prep?
    test_prep_benefit = 0.0
    df_prep = df_current.copy()
    if current_profile["test preparation course"] == "none":
        df_prep["test preparation course"] = "completed"
        df_prep_eng = engineer_features(df_prep)
        t_prep = preprocessor.transform(df_prep_eng)
        pred_with_prep = float(np.clip(model.predict(t_prep)[0], 0, 100))
        test_prep_benefit = round(pred_with_prep - current_pred, 1)
    else:
        pred_with_prep = current_pred
        
    # 3. Strategy B: Balanced Subject Improvement Plan
    req_reading = current_profile["reading score"]
    req_writing = current_profile["writing score"]
    
    if score_gap > 0:
        best_r = current_profile["reading score"]
        best_w = current_profile["writing score"]
        
        for delta in range(0, 50):
            test_r = min(100, current_profile["reading score"] + delta)
            test_w = min(100, current_profile["writing score"] + delta)
            
            df_test = df_prep.copy()
            df_test["reading score"] = test_r
            df_test["writing score"] = test_w
            df_test_eng = engineer_features(df_test)
            
            t_test = preprocessor.transform(df_test_eng)
            p_val = float(np.clip(model.predict(t_test)[0], 0, 100))
            
            if p_val >= target_score or (test_r == 100 and test_w == 100):
                best_r = test_r
                best_w = test_w
                break
                
        req_reading = best_r
        req_writing = best_w
        
    # 4. Feasibility Classification
    if score_gap <= 0:
        feasibility = "Already Achieved"
        badge_color = "#059669"
        advice = "The student's current profile is already projected to meet or exceed this target!"
    elif score_gap <= 6:
        feasibility = "Easily Achievable (Minor Effort)"
        badge_color = "#10B981"
        advice = "A slight boost in exam preparation or a few additional reading practice sessions will bridge this gap."
    elif score_gap <= 15:
        feasibility = "Moderate Effort Required"
        badge_color = "#3B82F6"
        advice = "Achievable with dedicated weekly practice, completing the test preparation course, and improving reading scores."
    else:
        feasibility = "Intensive Academic Intervention Needed"
        badge_color = "#DC2626"
        advice = "Requires comprehensive tutoring, daily reading/writing drills, and full test preparation support."
        
    return {
        "current_predicted_math": current_pred,
        "target_score": target_score,
        "score_gap": score_gap,
        "test_prep_benefit": test_prep_benefit,
        "required_reading_score": req_reading,
        "required_writing_score": req_writing,
        "reading_delta": req_reading - current_profile["reading score"],
        "writing_delta": req_writing - current_profile["writing score"],
        "feasibility": feasibility,
        "badge_color": badge_color,
        "advice": advice
    }
