import os
import pandas as pd
import numpy as np
import joblib

def test_inference():
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    model = joblib.load("artifacts/best_model.joblib")

    test_student = pd.DataFrame({
        "gender": ["female"],
        "race/ethnicity": ["group C"],
        "parental level of education": ["bachelor's degree"],
        "lunch": ["standard"],
        "test preparation course": ["completed"],
        "reading score": [85],
        "writing score": [88]
    })

    transformed = preprocessor.transform(test_student)
    pred_math = model.predict(transformed)[0]
    print(f"[TEST INFERENCE SUCCESS]")
    print(f"Sample Student Input:")
    print(test_student)
    print(f"\nPredicted Math Score: {pred_math:.2f} / 100")
    print(f"3-Subject Average: {(pred_math + 85 + 88) / 3:.2f} / 100")

if __name__ == "__main__":
    test_inference()
