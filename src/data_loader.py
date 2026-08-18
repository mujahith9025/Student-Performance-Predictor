import os
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Student_Performance.csv")

TARGET_COLUMN = "Performance_Index"
NUMERICAL_FEATURES = [
    "Hours_Studied",
    "Previous_Score",
    "Sleep_Hours",
    "Sample_Question_Papers_Practiced"
]
CATEGORICAL_FEATURES = [
    "Extracurricular_Activities"
]

def load_student_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads and cleans the Student Performance dataset from CSV.
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
