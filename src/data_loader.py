import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "Student_Performance.csv")

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

SUBJECT_METADATA = {
    "General Academics": {
        "filename": "Student_Performance.csv",
        "icon": "🌐",
        "badge": "General Baseline",
        "description": "Cross-disciplinary benchmark evaluating general academic performance across 10,000 real student records.",
        "focus": "Balanced study and previous test performance."
    },
    "Mathematics & Statistics": {
        "filename": "Student_Performance_Math.csv",
        "icon": "📐",
        "badge": "Quantitative & Logic",
        "description": "Rigorous quantitative performance emphasizing active problem solving, mock practice sets, and study depth.",
        "focus": "High impact on Sample Question Papers and Practice Density."
    },
    "Science & Physics": {
        "filename": "Student_Performance_Science.csv",
        "icon": "🔬",
        "badge": "STEM & Laboratory",
        "description": "STEM subjects evaluating conceptual mastery, experimental analysis, and balanced cognitive sleep recovery.",
        "focus": "High impact on Previous Foundation Score and Sleep-Study balance."
    },
    "Humanities & Literature": {
        "filename": "Student_Performance_Humanities.csv",
        "icon": "📖",
        "badge": "Arts & Language",
        "description": "Language, social sciences, and arts with strong correlation to extracurricular debate, reading, and steady preparation.",
        "focus": "High impact on Extracurriculars and consistent daily study."
    }
}

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

def create_subject_datasets_if_needed():
    """
    Ensures subject-specific calibrated datasets exist in the data directory based on the Kaggle baseline.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DEFAULT_DATA_PATH):
        return
        
    base_df = pd.read_csv(DEFAULT_DATA_PATH)
    base_df.columns = [col.strip().replace(" ", "_") for col in base_df.columns]
    
    np.random.seed(42)
    
    # 1. Mathematics: Stronger boost from Sample Question Papers & Practice Density
    math_path = os.path.join(DATA_DIR, "Student_Performance_Math.csv")
    if not os.path.exists(math_path):
        math_df = base_df.copy()
        extra_math = (math_df["Sample_Question_Papers_Practiced"] * 0.8) + (math_df["Hours_Studied"] * 0.5) - 3.0 + np.random.normal(0, 1.2, len(math_df))
        math_df["Performance_Index"] = np.clip(np.round(math_df["Performance_Index"] * 0.95 + extra_math, 1), 10.0, 100.0)
        math_df.to_csv(math_path, index=False)
        print(f"[+] Generated Mathematics dataset at: {math_path}")

    # 2. Science & Physics: Stronger boost from Previous Score & Sleep Quality
    science_path = os.path.join(DATA_DIR, "Student_Performance_Science.csv")
    if not os.path.exists(science_path):
        science_df = base_df.copy()
        extra_sci = (science_df["Previous_Score"] * 0.05) + (science_df["Sleep_Hours"] * 0.4) - 2.5 + np.random.normal(0, 1.3, len(science_df))
        science_df["Performance_Index"] = np.clip(np.round(science_df["Performance_Index"] * 0.96 + extra_sci, 1), 10.0, 100.0)
        science_df.to_csv(science_path, index=False)
        print(f"[+] Generated Science dataset at: {science_path}")

    # 3. Humanities & Literature: Stronger boost from Extracurriculars and Reading/Sleep
    humanities_path = os.path.join(DATA_DIR, "Student_Performance_Humanities.csv")
    if not os.path.exists(humanities_path):
        hum_df = base_df.copy()
        ec_bonus = np.where(hum_df["Extracurricular_Activities"] == "Yes", 3.2, -1.0)
        extra_hum = ec_bonus + (hum_df["Hours_Studied"] * 0.4) + (hum_df["Sleep_Hours"] * 0.3) - 2.0 + np.random.normal(0, 1.4, len(hum_df))
        hum_df["Performance_Index"] = np.clip(np.round(hum_df["Performance_Index"] * 0.94 + extra_hum, 1), 10.0, 100.0)
        hum_df.to_csv(humanities_path, index=False)
        print(f"[+] Generated Humanities dataset at: {humanities_path}")

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

def load_subject_data(subject_name: str = "General Academics") -> pd.DataFrame:
    """
    Loads dataset for a specific academic subject.
    """
    create_subject_datasets_if_needed()
    meta = SUBJECT_METADATA.get(subject_name, SUBJECT_METADATA["General Academics"])
    filepath = os.path.join(DATA_DIR, meta["filename"])
    return load_student_data(filepath)

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
