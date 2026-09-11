import os
import io
import random
import pandas as pd
import numpy as np

# Realistic English student names for synthetic cohorts
FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Sophia", "James", "Isabella", "Lucas", "Mia", "Ethan",
    "Ava", "Mason", "Charlotte", "Oliver", "Amelia", "Elijah", "Harper", "Aiden", "Evelyn", "Alexander",
    "Abigail", "Henry", "Emily", "Sebastian", "Ella", "Benjamin", "Elizabeth", "Daniel", "Camila", "Matthew",
    "Luna", "Jackson", "Sofia", "David", "Avery", "Joseph", "Mila", "Samuel", "Aria", "John",
    "Scarlett", "Luke", "Penelope", "Carter", "Chloe", "Anthony", "Layla", "Dylan", "Victoria", "Gabriel",
    "Grace", "Leo", "Zoey", "Isaac", "Nora", "Andrew", "Riley", "Julian", "Lily", "Joshua",
    "Eleanor", "Christopher", "Hannah", "Nathan", "Lillian", "Caleb", "Addison", "Ryan", "Aubrey", "Adrian",
    "Ellie", "Christian", "Stella", "Jaxon", "Natalie", "Aaron", "Zoe", "Eli", "Leah", "Landon"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
]

def generate_synthetic_classroom(n_students=50, cohort_type="balanced", seed=42):
    """
    Generates a realistic synthetic classroom dataset with student_id, student_name,
    and all 7 required academic and socioeconomic features.
    
    cohort_type options:
    - 'balanced': Normal distributed class (mix of high, mid, and low performers)
    - 'at_risk_focus': Higher proportion of struggling students requiring intervention
    - 'honors_advanced': High achieving students striving for distinction
    - 'large_cohort': Diverse multi-tier cohort
    """
    random.seed(seed)
    np.random.seed(seed)
    
    genders = ['female', 'male']
    race_ethnicity = ['group A', 'group B', 'group C', 'group D', 'group E']
    race_weights = [0.09, 0.19, 0.32, 0.26, 0.14]
    
    parental_education = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    
    if cohort_type == "honors_advanced":
        edu_weights = [0.03, 0.07, 0.20, 0.25, 0.30, 0.15]
        lunch_weights = [0.85, 0.15]
        prep_weights = [0.20, 0.80]
        base_mean = 84.0
        base_std = 7.5
    elif cohort_type == "at_risk_focus":
        edu_weights = [0.35, 0.30, 0.20, 0.10, 0.04, 0.01]
        lunch_weights = [0.30, 0.70]
        prep_weights = [0.80, 0.20]
        base_mean = 48.0
        base_std = 9.0
    else:  # balanced or large_cohort
        edu_weights = [0.18, 0.20, 0.23, 0.22, 0.12, 0.05]
        lunch_weights = [0.65, 0.35]
        prep_weights = [0.55, 0.45]
        base_mean = 68.0
        base_std = 12.0
        
    edu_boost = {
        'some high school': -4,
        'high school': -2,
        'some college': 0,
        "associate's degree": 2,
        "bachelor's degree": 5,
        "master's degree": 8
    }
    lunch_boost = {'standard': 4, 'free/reduced': -5}
    prep_boost = {'completed': 6, 'none': -3}
    
    records = []
    used_names = set()
    
    for i in range(1, n_students + 1):
        # Generate Unique Name
        while True:
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            full_name = f"{fn} {ln}"
            if full_name not in used_names or len(used_names) >= len(FIRST_NAMES) * len(LAST_NAMES):
                used_names.add(full_name)
                break
                
        student_id = f"STU-2026-{1000 + i:04d}"
        gender = random.choices(genders, weights=[0.51, 0.49])[0]
        race = random.choices(race_ethnicity, weights=race_weights)[0]
        edu = random.choices(parental_education, weights=edu_weights)[0]
        lunch = random.choices(lunches := ['standard', 'free/reduced'], weights=lunch_weights)[0]
        prep = random.choices(preps := ['none', 'completed'], weights=prep_weights)[0]
        
        base = np.random.normal(base_mean, base_std)
        adj = edu_boost[edu] + lunch_boost[lunch] + prep_boost[prep]
        
        g_read = -2.5 if gender == 'male' else 2.5
        g_write = -3.5 if gender == 'male' else 3.5
        
        reading = int(np.clip(round(base + adj + g_read + np.random.normal(0, 5)), 15, 100))
        writing = int(np.clip(round((reading * 0.65) + (base + adj + g_write) * 0.35 + np.random.normal(0, 3)), 15, 100))
        
        records.append({
            "student_id": student_id,
            "student_name": full_name,
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": edu,
            "lunch": lunch,
            "test preparation course": prep,
            "reading score": reading,
            "writing score": writing
        })
        
    df = pd.DataFrame(records)
    return df

def generate_and_save_sample_files():
    """
    Generates sample classroom CSV files for testing bulk predictions and saves them to data/sample_classrooms/
    """
    output_dir = os.path.join("data", "sample_classrooms")
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_generate = [
        ("sample_classroom_balanced_50.csv", 50, "balanced", 42),
        ("sample_classroom_large_100.csv", 100, "large_cohort", 101),
        ("sample_classroom_at_risk_focus_40.csv", 40, "at_risk_focus", 77),
        ("sample_classroom_honors_35.csv", 35, "honors_advanced", 99),
        ("sample_classroom_mixed_200.csv", 200, "balanced", 123)
    ]
    
    generated_paths = []
    for filename, count, cohort_type, seed in files_to_generate:
        df = generate_synthetic_classroom(n_students=count, cohort_type=cohort_type, seed=seed)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        generated_paths.append(filepath)
        print(f" Generated: {filepath} ({len(df)} rows)")
        
    return generated_paths

if __name__ == "__main__":
    generate_and_save_sample_files()
