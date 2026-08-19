import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    load_student_data,
    load_subject_data,
    engineer_features,
    get_feature_target_split,
    split_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    SUBJECT_METADATA
)

def test_load_student_data():
    """Verify that Kaggle student dataset loads properly with required columns."""
    df = load_student_data()
    assert df is not None
    assert len(df) >= 9000
    expected_cols = [
        "Hours_Studied", "Previous_Score", "Extracurricular_Activities",
        "Sleep_Hours", "Sample_Question_Papers_Practiced", "Performance_Index"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert df["Performance_Index"].min() >= 10.0
    assert df["Performance_Index"].max() <= 100.0

def test_load_all_subjects():
    """Verify that datasets for all 4 academic subjects load cleanly."""
    for subject_name in SUBJECT_METADATA.keys():
        df = load_subject_data(subject_name)
        assert df is not None, f"Failed to load subject: {subject_name}"
        assert len(df) > 0
        assert "Performance_Index" in df.columns

def test_engineer_features_calculations():
    """Test custom feature engineering equations."""
    sample_raw = pd.DataFrame([{
        "Hours_Studied": 6,
        "Previous_Score": 80,
        "Sleep_Hours": 8,
        "Sample_Question_Papers_Practiced": 4,
        "Extracurricular_Activities": "Yes"
    }])
    
    eng = engineer_features(sample_raw)
    
    # 1. Study_to_Sleep_Ratio = 6 / 8 = 0.75
    assert "Study_to_Sleep_Ratio" in eng.columns
    assert pytest.approx(eng["Study_to_Sleep_Ratio"].iloc[0], 0.01) == 0.75
    
    # 2. Practice_Density = 4 / (6 + 1) = 4 / 7 ≈ 0.5714
    assert "Practice_Density" in eng.columns
    assert pytest.approx(eng["Practice_Density"].iloc[0], 0.01) == (4 / 7)
    
    # 3. Study_Effort_Score = (0.6 * 6) + (0.4 * 4) = 3.6 + 1.6 = 5.2
    assert "Study_Effort_Score" in eng.columns
    assert pytest.approx(eng["Study_Effort_Score"].iloc[0], 0.01) == 5.2

def test_engineer_features_zero_sleep_safety():
    """Ensure zero or negative sleep hours does not throw DivisionByZero error."""
    zero_sleep_df = pd.DataFrame([{
        "Hours_Studied": 5,
        "Previous_Score": 70,
        "Sleep_Hours": 0,
        "Sample_Question_Papers_Practiced": 2,
        "Extracurricular_Activities": "No"
    }])
    eng = engineer_features(zero_sleep_df)
    assert not np.isinf(eng["Study_to_Sleep_Ratio"].iloc[0])
    assert not np.isnan(eng["Study_to_Sleep_Ratio"].iloc[0])

def test_get_feature_target_split_and_train_test():
    """Test train-test split dimensions and stratification."""
    df = load_student_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "Study_to_Sleep_Ratio" in X_train.columns
