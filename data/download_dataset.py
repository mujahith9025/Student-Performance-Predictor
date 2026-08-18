import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, "Student_Performance.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_batch_test.csv")

# Verified mirror of Kaggle 'Student Performance (Multiple Linear Regression)' dataset (10,000 records)
DATA_URL = "https://raw.githubusercontent.com/Datakortex/Datasets/refs/heads/main/student_performance.csv"

def download_and_prepare_data():
    """
    Downloads the real Kaggle Student Performance dataset, validates its columns,
    and prepares a sample batch CSV for testing batch predictions in the web app.
    """
    print(f"[*] Downloading real Kaggle Student Performance dataset from:\n    {DATA_URL}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download file
    urllib.request.urlretrieve(DATA_URL, DATASET_PATH)
    print(f"[+] Download complete: {DATASET_PATH}")
    
    # Load and validate
    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Dataset successfully loaded with shape: {df.shape}")
    print(f"[*] Columns detected: {list(df.columns)}")
    
    # Standardize column naming if necessary
    # Expected: Hours_Studied, Previous_Score, Extracurricular_Activities, Sleep_Hours, Sample_Question_Papers_Practiced, Performance_Index
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df.to_csv(DATASET_PATH, index=False)
    
    # Create sample batch test file (20 records without the target variable)
    sample_batch = df.sample(n=20, random_state=42).copy()
    if "Performance_Index" in sample_batch.columns:
        sample_batch = sample_batch.drop(columns=["Performance_Index"])
    
    sample_batch.to_csv(SAMPLE_BATCH_PATH, index=False)
    print(f"[+] Created sample batch test file with 20 rows: {SAMPLE_BATCH_PATH}")
    
    print("\n--- Dataset Summary ---")
    print(df.describe(include='all'))
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n[OK] Data layer ready!")

if __name__ == "__main__":
    download_and_prepare_data()
