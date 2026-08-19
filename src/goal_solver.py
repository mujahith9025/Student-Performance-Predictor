import os
import sys
import numpy as np
import pandas as pd
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

for p in [CURRENT_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data_loader import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
except ImportError:
    from src.data_loader import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def solve_target_score(pipeline, previous_score: float, current_hours: float, current_sleep: float, current_papers: int, extracurricular: str, target_score: float) -> dict:
    """
    Inverse Goal Solver / Reverse Optimizer:
    Finds the optimal, realistic, and healthy study habit combinations required to achieve a desired Target Score.
    """
    # 1. Generate search grid across feasible healthy ranges
    hours_range = np.linspace(1.0, 9.0, 17) # 1.0, 1.5, 2.0, ..., 9.0
    papers_range = np.arange(0, 10, 1)      # 0, 1, 2, ..., 9
    sleep_range = np.array([5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]) # Healthy sleep bounds
    
    grid = []
    for h in hours_range:
        for p in papers_range:
            for s in sleep_range:
                grid.append({
                    "Hours_Studied": float(h),
                    "Previous_Score": float(previous_score),
                    "Sleep_Hours": float(s),
                    "Sample_Question_Papers_Practiced": int(p),
                    "Extracurricular_Activities": extracurricular
                })
                
    df_grid = pd.DataFrame(grid)
    df_eng = engineer_features(df_grid)
    
    preds = pipeline.predict(df_eng)
    df_grid["Predicted_Score"] = np.round(np.clip(preds, 10.0, 100.0), 1)
    
    # Baseline current score
    curr_df = pd.DataFrame([{
        "Hours_Studied": float(current_hours),
        "Previous_Score": float(previous_score),
        "Sleep_Hours": float(current_sleep),
        "Sample_Question_Papers_Practiced": int(current_papers),
        "Extracurricular_Activities": extracurricular
    }])
    current_pred = round(float(pipeline.predict(engineer_features(curr_df))[0]), 1)
    
    max_achievable = float(df_grid["Predicted_Score"].max())
    min_achievable = float(df_grid["Predicted_Score"].min())
    
    # If target is above maximum possible in 1 cycle
    if target_score > max_achievable:
        # Find best possible configuration
        best_row = df_grid.sort_values(by="Predicted_Score", ascending=False).iloc[0]
        max_pathway = {
            "name": "🚀 Maximum Performance Peak",
            "tag": "Max Ceiling",
            "description": f"Pushes habits to their maximum academic ceiling ({max_achievable:.1f} pts). Reaching {target_score:.1f} will require building on this score in the subsequent testing cycle.",
            "required_hours": float(best_row["Hours_Studied"]),
            "required_papers": int(best_row["Sample_Question_Papers_Practiced"]),
            "required_sleep": float(best_row["Sleep_Hours"]),
            "predicted_score": float(best_row["Predicted_Score"]),
            "delta_hours": round(float(best_row["Hours_Studied"] - current_hours), 1),
            "delta_papers": int(best_row["Sample_Question_Papers_Practiced"] - current_papers),
            "weekly_study_hours": round(float(best_row["Hours_Studied"] * 7), 1),
            "confidence": "Ceiling Limit"
        }
        return {
            "feasible": False,
            "target_score": target_score,
            "current_pred": current_pred,
            "max_achievable": max_achievable,
            "gap": round(target_score - current_pred, 1),
            "message": f"Target score of {target_score:.1f} exceeds the maximum single-cycle ceiling ({max_achievable:.1f} pts) possible from a previous foundation of {previous_score:.0f} marks. The optimal roadmap below gets you to the maximum {max_achievable:.1f} pts.",
            "pathways": [max_pathway]
        }
        
    # Filter configurations meeting or slightly exceeding target
    candidates = df_grid[df_grid["Predicted_Score"] >= target_score].copy()
    if len(candidates) == 0:
        candidates = df_grid[df_grid["Predicted_Score"] >= (target_score - 1.0)].copy()
        
    # Pathway 1: Balanced Roadmap (Optimal 7.5 hrs sleep, minimum extreme fatigue)
    candidates["sleep_penalty"] = abs(candidates["Sleep_Hours"] - 7.5)
    candidates["balance_cost"] = (candidates["sleep_penalty"] * 3.0) + candidates["Hours_Studied"] + (candidates["Sample_Question_Papers_Practiced"] * 0.4)
    balanced_row = candidates.sort_values(by=["balance_cost", "Predicted_Score"]).iloc[0]
    
    # Pathway 2: Practice-Intensive Roadmap (High Mock Papers, moderate hours)
    practice_row = candidates.sort_values(by=["Sample_Question_Papers_Practiced", "Hours_Studied"], ascending=[False, True]).iloc[0]
    
    # Pathway 3: Time-Efficient / Fast Pathway (Minimum Study Hours)
    time_row = candidates.sort_values(by=["Hours_Studied", "Sample_Question_Papers_Practiced"]).iloc[0]
    
    pathways = [
        {
            "name": "⚖️ Balanced & Sustainable Mastery",
            "tag": "Recommended",
            "description": "Optimal balance between focused daily study, mock testing, and restorative 7-8 hours of sleep.",
            "required_hours": float(balanced_row["Hours_Studied"]),
            "required_papers": int(balanced_row["Sample_Question_Papers_Practiced"]),
            "required_sleep": float(balanced_row["Sleep_Hours"]),
            "predicted_score": float(balanced_row["Predicted_Score"]),
            "delta_hours": round(float(balanced_row["Hours_Studied"] - current_hours), 1),
            "delta_papers": int(balanced_row["Sample_Question_Papers_Practiced"] - current_papers),
            "weekly_study_hours": round(float(balanced_row["Hours_Studied"] * 7), 1),
            "confidence": "High (Sustainable routine)"
        },
        {
            "name": "📄 Mock-Test Sprint Pathway",
            "tag": "High Active Recall",
            "description": "Accelerates score improvement by maximizing sample question papers and practice density.",
            "required_hours": float(practice_row["Hours_Studied"]),
            "required_papers": int(practice_row["Sample_Question_Papers_Practiced"]),
            "required_sleep": float(practice_row["Sleep_Hours"]),
            "predicted_score": float(practice_row["Predicted_Score"]),
            "delta_hours": round(float(practice_row["Hours_Studied"] - current_hours), 1),
            "delta_papers": int(practice_row["Sample_Question_Papers_Practiced"] - current_papers),
            "weekly_study_hours": round(float(practice_row["Hours_Studied"] * 7), 1),
            "confidence": "Medium-High (Practice heavy)"
        },
        {
            "name": "⚡ Time-Efficient Pathway",
            "tag": "Minimal Daily Time",
            "description": "Achieves the goal with the fewest daily study hours by prioritizing high-yield testing.",
            "required_hours": float(time_row["Hours_Studied"]),
            "required_papers": int(time_row["Sample_Question_Papers_Practiced"]),
            "required_sleep": float(time_row["Sleep_Hours"]),
            "predicted_score": float(time_row["Predicted_Score"]),
            "delta_hours": round(float(time_row["Hours_Studied"] - current_hours), 1),
            "delta_papers": int(time_row["Sample_Question_Papers_Practiced"] - current_papers),
            "weekly_study_hours": round(float(time_row["Hours_Studied"] * 7), 1),
            "confidence": "High Efficiency"
        }
    ]
    
    return {
        "feasible": True,
        "target_score": target_score,
        "current_pred": current_pred,
        "max_achievable": max_achievable,
        "min_achievable": min_achievable,
        "gap": round(target_score - current_pred, 1),
        "pathways": pathways
    }
