import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Student_Performance.csv")

TARGET_COLUMN = "Performance_Index"

# Base numerical features
BASE_NUMERICAL_FEATURES = [
    "Hours_Studied",
    "Previous_Score",
    "Sleep_Hours",
    "Sample_Question_Papers_Practiced"
]

# Engineered numerical features
ENGINEERED_NUMERICAL_FEATURES = [
    "Study_to_Sleep_Ratio",
    "Practice_Density",
    "Study_Effort_Score"
]

NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_FEATURES = [
    "Extracurricular_Activities"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom domain-specific feature engineering to the Student Performance data:
    1. Study_to_Sleep_Ratio: Study intensity vs rest balance (Hours_Studied / Sleep_Hours).
    2. Practice_Density: Mock test volume normalized by study hours (Sample_Papers / (Hours + 1)).
    3. Study_Effort_Score: Weighted academic effort index combining study time and mock tests.
    """
    df_out = df.copy()
    
    # 1. Study-to-Sleep Ratio (Work/Rest Balance)
    sleep_safe = df_out["Sleep_Hours"].replace(0, 1)
    df_out["Study_to_Sleep_Ratio"] = np.round(df_out["Hours_Studied"] / sleep_safe, 3)
    
    # 2. Practice Density (Mock Papers per Study Hour)
    df_out["Practice_Density"] = np.round(
        df_out["Sample_Question_Papers_Practiced"] / (df_out["Hours_Studied"] + 1), 3
    )
    
    # 3. Composite Study Effort Score (60% study time + 40% practical testing)
    df_out["Study_Effort_Score"] = np.round(
        (df_out["Hours_Studied"] * 0.6) + (df_out["Sample_Question_Papers_Practiced"] * 0.4), 3
    )
    
    return df_out

def load_student_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads, cleans, and applies custom feature engineering to the Student Performance dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please run data/download_dataset.py first."
        )
    
    df = pd.read_csv(filepath)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    
    # Drop exact duplicates if any
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[*] Removed {initial_len - len(df)} duplicate rows.")
        
    # Apply Feature Engineering
    df = engineer_features(df)
    return df

def get_feature_target_split(df: pd.DataFrame):
    """
    Separates feature matrix X and target series y.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
        
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits feature matrix and target series into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
