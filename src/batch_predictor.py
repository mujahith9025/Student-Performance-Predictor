import io
import os
import pandas as pd
import numpy as np
from src.advanced_feature_engineering import engineer_features
from src.sample_generator import generate_synthetic_classroom

# Column standard name aliases mapping
COLUMN_ALIASES = {
    "reading_score": "reading score",
    "reading": "reading score",
    "writing_score": "writing score",
    "writing": "writing score",
    "math_score": "math score",
    "math": "math score",
    "parental_level_of_education": "parental level of education",
    "parent_education": "parental level of education",
    "parental_education": "parental level of education",
    "race_ethnicity": "race/ethnicity",
    "ethnicity": "race/ethnicity",
    "test_preparation_course": "test preparation course",
    "test_prep": "test preparation course",
    "study_hours": "weekly_study_hours",
    "attendance": "attendance_rate",
    "screen_time": "daily_screen_time_hours",
    "failures": "past_failures",
    "previous_score": "previous_term_score",
    "prior_score": "previous_term_score",
    "prior_term_score": "previous_term_score"
}

ALL_18_DEFAULTS = {
    "gender": "female",
    "race/ethnicity": "group C",
    "parental level of education": "some college",
    "lunch": "standard",
    "test preparation course": "none",
    "reading score": 65.0,
    "writing score": 65.0,
    "internet_access": "yes",
    "extracurricular_activities": "no",
    "tutoring_support": "none",
    "study_method": "spaced_repetition",
    "parental_involvement": "medium",
    "previous_term_score": 65.0,
    "attendance_rate": 85.0,
    "weekly_study_hours": 12.0,
    "sleep_hours_per_day": 7.5,
    "daily_screen_time_hours": 3.0,
    "past_failures": 0
}

def generate_sample_csv_template():
    """
    Generates a starter CSV template with 10 student records containing all 18 features.
    """
    sample_data = {
        "student_id": [f"STU-2026-{100+i}" for i in range(1, 11)],
        "student_name": [
            "Emma Watson", "Liam Smith", "Olivia Brown", "Noah Davis", "Sophia Wilson",
            "James Taylor", "Isabella Anderson", "Lucas Thomas", "Mia Jackson", "Ethan White"
        ],
        "gender": ["female", "male", "female", "male", "female", "male", "female", "male", "female", "male"],
        "race/ethnicity": ["group B", "group C", "group A", "group D", "group E", "group C", "group B", "group D", "group C", "group A"],
        "parental level of education": [
            "bachelor's degree", "some college", "high school", "master's degree", "associate's degree",
            "some high school", "some college", "bachelor's degree", "high school", "some high school"
        ],
        "lunch": ["standard", "free/reduced", "standard", "standard", "standard", "free/reduced", "standard", "standard", "free/reduced", "free/reduced"],
        "test preparation course": ["completed", "none", "none", "completed", "completed", "none", "completed", "completed", "none", "none"],
        "internet_access": ["yes", "yes", "yes", "yes", "yes", "no", "yes", "yes", "no", "no"],
        "extracurricular_activities": ["yes", "no", "no", "yes", "yes", "no", "yes", "yes", "no", "no"],
        "tutoring_support": ["peer_tutoring", "none", "none", "private_tutor", "peer_tutoring", "none", "peer_tutoring", "private_tutor", "none", "none"],
        "study_method": [
            "active_problem_solving", "passive_reading", "group_study", "active_problem_solving",
            "spaced_repetition", "passive_reading", "spaced_repetition", "active_problem_solving",
            "group_study", "passive_reading"
        ],
        "parental_involvement": ["high", "medium", "low", "high", "high", "low", "high", "high", "medium", "low"],
        "previous_term_score": [86, 56, 60, 91, 84, 40, 76, 88, 50, 36],
        "attendance_rate": [94.5, 78.0, 84.0, 96.0, 92.0, 68.5, 89.0, 95.0, 74.0, 62.0],
        "weekly_study_hours": [18.5, 8.0, 11.0, 22.0, 16.5, 5.0, 14.0, 20.0, 7.5, 4.0],
        "sleep_hours_per_day": [7.8, 6.5, 7.0, 8.0, 7.5, 5.2, 7.6, 8.2, 6.0, 5.0],
        "daily_screen_time_hours": [2.2, 4.8, 3.5, 1.8, 2.5, 6.0, 2.8, 1.5, 5.2, 6.5],
        "past_failures": [0, 1, 0, 0, 0, 2, 0, 0, 1, 3],
        "reading score": [88, 54, 62, 92, 85, 42, 78, 89, 49, 38],
        "writing score": [90, 52, 60, 94, 86, 39, 82, 91, 46, 35]
    }
    df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def load_or_create_sample_cohort(cohort_type="balanced", n_students=50, seed=42):
    file_map = {
        "balanced_50": os.path.join("data", "sample_classrooms", "sample_classroom_balanced_50.csv"),
        "large_100": os.path.join("data", "sample_classrooms", "sample_classroom_large_100.csv"),
        "at_risk_40": os.path.join("data", "sample_classrooms", "sample_classroom_at_risk_focus_40.csv"),
        "honors_35": os.path.join("data", "sample_classrooms", "sample_classroom_honors_35.csv"),
        "mixed_200": os.path.join("data", "sample_classrooms", "sample_classroom_mixed_200.csv")
    }
    
    if cohort_type in file_map and os.path.exists(file_map[cohort_type]):
        df = pd.read_csv(file_map[cohort_type])
        return df
        
    cohort_map = {
        "balanced_50": ("balanced", 50),
        "large_100": ("large_cohort", 100),
        "at_risk_40": ("at_risk_focus", 40),
        "honors_35": ("honors_advanced", 35),
        "mixed_200": ("balanced", 200)
    }
    c_type, count = cohort_map.get(cohort_type, (cohort_type, n_students))
    df = generate_synthetic_classroom(n_students=count, cohort_type=c_type, seed=seed)
    return df

def process_batch_predictions(df, preprocessor, reg_model, clf_model):
    """
    Processes an entire classroom DataFrame across all 18 features and returns the enriched DataFrame + summary analytics.
    Gracefully cleans column headers and imputes defaults for missing columns to prevent user crashes.
    """
    if df is None or len(df) == 0:
        raise ValueError("Uploaded CSV is empty. Please upload a file with student rows.")

    processed_df = df.copy()

    # 1. Normalize Column Headers
    rename_dict = {}
    for col in processed_df.columns:
        clean_col = str(col).strip().lower()
        if clean_col in COLUMN_ALIASES:
            rename_dict[col] = COLUMN_ALIASES[clean_col]
        else:
            rename_dict[col] = str(col).strip()
    processed_df = processed_df.rename(columns=rename_dict)

    # 2. Impute IDs and Names if missing
    if "student_id" not in processed_df.columns:
        processed_df["student_id"] = [f"STU-{1001 + i}" for i in range(len(processed_df))]
    if "student_name" not in processed_df.columns:
        processed_df["student_name"] = [f"Student {i + 1}" for i in range(len(processed_df))]

    # 3. Impute defaults for any missing features
    for col, default_val in ALL_18_DEFAULTS.items():
        if col not in processed_df.columns:
            processed_df[col] = default_val
        else:
            # Handle empty / NaN values in existing columns
            if pd.api.types.is_numeric_dtype(type(default_val)):
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(default_val)
            else:
                processed_df[col] = processed_df[col].fillna(default_val).astype(str)

    # Coerce numeric bounds
    numeric_bounds = {
        "reading score": (0.0, 100.0),
        "writing score": (0.0, 100.0),
        "previous_term_score": (0.0, 100.0),
        "attendance_rate": (0.0, 100.0),
        "weekly_study_hours": (0.0, 60.0),
        "sleep_hours_per_day": (1.0, 14.0),
        "daily_screen_time_hours": (0.0, 24.0),
        "past_failures": (0, 10)
    }
    for col, (low, high) in numeric_bounds.items():
        if col in processed_df.columns:
            processed_df[col] = np.clip(pd.to_numeric(processed_df[col], errors='coerce').fillna(ALL_18_DEFAULTS[col]), low, high)

    # Apply feature engineering
    feature_df_eng = engineer_features(processed_df)
    transformed = preprocessor.transform(feature_df_eng)

    # 1. Regression: Predict Math Marks
    predicted_math = reg_model.predict(transformed)
    predicted_math = np.clip(predicted_math, 0, 100).round(1)
    processed_df["Predicted_Math_Score"] = predicted_math

    # 2. Cumulative 3-Subject Average
    reading = processed_df["reading score"]
    writing = processed_df["writing score"]
    cumulative_avg = ((predicted_math + reading + writing) / 3.0).round(1)
    processed_df["Cumulative_3Subject_Avg"] = cumulative_avg

    # 3. Letter Grade Assignment
    def assign_grade(score):
        if score >= 90: return "A+ (Outstanding)"
        elif score >= 80: return "A (Excellent)"
        elif score >= 70: return "B (Good)"
        elif score >= 60: return "C (Satisfactory)"
        elif score >= 50: return "D (Pass)"
        else: return "F (Needs Remedial)"

    processed_df["Predicted_Grade"] = cumulative_avg.apply(assign_grade)

    # 4. Classification: Pass Probability & Risk Tiers
    if clf_model is not None:
        pass_probs = (clf_model.predict_proba(transformed)[:, 1] * 100.0).round(1)
        pass_flags = clf_model.predict(transformed)
    else:
        pass_probs = np.where(predicted_math >= 50, 95.0, 25.0)
        pass_flags = np.where(predicted_math >= 50, 1, 0)

    processed_df["Pass_Probability_Pct"] = pass_probs
    processed_df["Pass_Status"] = np.where(pass_flags == 1, "Pass", "At-Risk / Fail")

    def assign_risk_tier(prob):
        if prob >= 80: return "Safe / Low Risk"
        elif prob >= 50: return "Moderate Risk"
        else: return "🚨 High Academic Risk"

    processed_df["Risk_Tier"] = pd.Series(pass_probs).apply(assign_risk_tier)

    # 5. Prescriptive Solution Tagging for Each Student
    def assign_prescriptive_action(row):
        math_s = row["Predicted_Math_Score"]
        att = row.get("attendance_rate", 85.0)
        fails = row.get("past_failures", 0)
        prep_s = row.get("test preparation course", "none")
        screen_s = row.get("daily_screen_time_hours", 3.0)
        method_s = row.get("study_method", "spaced_repetition")
        prob = row["Pass_Probability_Pct"]
        
        if prob < 50 or math_s < 50 or fails >= 2:
            return "🚨 Intensive 1-on-1 Tutoring & Backlog Remediation"
        elif screen_s > 4.5:
            return "📱 Screen Time Reduction & Focus Blocks"
        elif method_s == "passive_reading":
            return "📖 Active Problem Solving & Recall Practice"
        elif att < 80.0:
            return "⏱️ Attendance Tracking & Habit Intervention"
        elif prep_s == "none" and math_s < 80:
            return "🎯 4-Week Exam Prep Course Enrollment"
        elif math_s >= 85:
            return "🏆 Honors & Olympiad Advancement"
        else:
            return "📈 Weekly Guided Problem Solving"

    processed_df["Prescribed_Intervention"] = processed_df.apply(assign_prescriptive_action, axis=1)

    # 6. Class-level Summary Analytics
    total_students = len(processed_df)
    pass_count = int(np.sum(pass_flags == 1))
    at_risk_count = total_students - pass_count
    pass_rate = round((pass_count / total_students) * 100, 1)
    class_avg_math = round(float(np.mean(predicted_math)), 1)
    class_avg_overall = round(float(np.mean(cumulative_avg)), 1)
    distinction_count = int(np.sum(cumulative_avg >= 80))

    summary = {
        "total_students": total_students,
        "pass_count": pass_count,
        "at_risk_count": at_risk_count,
        "pass_rate": pass_rate,
        "class_avg_math": class_avg_math,
        "class_avg_overall": class_avg_overall,
        "distinction_count": distinction_count
    }

    return processed_df, summary
