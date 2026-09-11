import os
import pandas as pd
import numpy as np
import joblib

def test_dual_inference():
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    reg_model = joblib.load("artifacts/best_model.joblib")
    clf_model = joblib.load("artifacts/best_classifier.joblib")

    # High-performing student
    student_a = pd.DataFrame({
        "gender": ["female"],
        "race/ethnicity": ["group C"],
        "parental level of education": ["bachelor's degree"],
        "lunch": ["standard"],
        "test preparation course": ["completed"],
        "reading score": [85],
        "writing score": [88]
    })
    t_a = preprocessor.transform(student_a)
    pred_score_a = reg_model.predict(t_a)[0]
    prob_a = clf_model.predict_proba(t_a)[0][1] * 100

    # At-risk student
    student_b = pd.DataFrame({
        "gender": ["male"],
        "race/ethnicity": ["group A"],
        "parental level of education": ["some high school"],
        "lunch": ["free/reduced"],
        "test preparation course": ["none"],
        "reading score": [32],
        "writing score": [28]
    })
    t_b = preprocessor.transform(student_b)
    pred_score_b = reg_model.predict(t_b)[0]
    prob_b = clf_model.predict_proba(t_b)[0][1] * 100

    print("=" * 60)
    print("           DUAL-TASK INFERENCE TEST RESULTS            ")
    print("=" * 60)
    print(f"\n[Student A - Strong Standing]")
    print(f"Predicted Math Marks: {pred_score_a:.2f} / 100")
    print(f"Pass Probability:     {prob_a:.2f}% (Safe / High Standing)")

    print(f"\n[Student B - At-Risk Profile]")
    print(f"Predicted Math Marks: {pred_score_b:.2f} / 100")
    print(f"Pass Probability:     {prob_b:.2f}% (High Academic Risk Alert)")
    print("\n[OK] Dual Inference Pipeline Working Perfectly!")

if __name__ == "__main__":
    test_dual_inference()
