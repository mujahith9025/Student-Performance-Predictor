import io
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "reading score",
    "writing score"
]

def generate_sample_csv_template():
    """
    Generates a sample CSV template with 10 student records for teachers to test bulk predictions.
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
        "reading score": [88, 54, 62, 92, 85, 42, 78, 89, 49, 38],
        "writing score": [90, 52, 60, 94, 86, 39, 82, 91, 46, 35]
    }
    df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def process_batch_predictions(df, preprocessor, reg_model, clf_model):
    """
    Processes an entire classroom DataFrame and returns the enriched DataFrame + summary analytics.
    """
    # Verify required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing_cols}")

    processed_df = df.copy()
    feature_df = processed_df[REQUIRED_COLUMNS].copy()

    # Preprocessing
    transformed = preprocessor.transform(feature_df)

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
        pass_probs = np.where(predicted_math >= 50, 90.0, 30.0)
        pass_flags = np.where(predicted_math >= 50, 1, 0)

    processed_df["Pass_Probability_Pct"] = pass_probs
    processed_df["Pass_Status"] = np.where(pass_flags == 1, "Pass", "At-Risk / Fail")

    def assign_risk_tier(prob):
        if prob >= 80: return "Safe / Low Risk"
        elif prob >= 50: return "Moderate Risk"
        else: return "🚨 High Academic Risk"

    processed_df["Risk_Tier"] = pd.Series(pass_probs).apply(assign_risk_tier)

    # 5. Class-level Summary Analytics
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
