import csv
import random
import numpy as np

def generate_full_dataset(n_samples=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    genders = ['female', 'male']
    race_ethnicity = ['group A', 'group B', 'group C', 'group D', 'group E']
    race_weights = [0.089, 0.190, 0.319, 0.262, 0.140]
    
    parental_education = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    edu_weights = [0.179, 0.196, 0.226, 0.222, 0.118, 0.059]
    
    lunches = ['standard', 'free/reduced']
    lunch_weights = [0.645, 0.355]
    
    test_preps = ['none', 'completed']
    prep_weights = [0.642, 0.358]
    
    internet_opts = ['yes', 'no']
    internet_weights = [0.82, 0.18]
    
    extra_opts = ['yes', 'no']
    extra_weights = [0.48, 0.52]
    
    tutoring_opts = ['none', 'peer_tutoring', 'private_tutor']
    tutoring_weights = [0.60, 0.25, 0.15]
    
    edu_boost = {
        'some high school': -4,
        'high school': -2,
        'some college': 0,
        "associate's degree": 2,
        "bachelor's degree": 5,
        "master's degree": 8
    }
    lunch_boost = {'standard': 3.5, 'free/reduced': -4.5}
    prep_boost = {'completed': 5.5, 'none': -2.5}
    internet_boost = {'yes': 2.0, 'no': -3.0}
    tutoring_boost = {'none': 0.0, 'peer_tutoring': 3.5, 'private_tutor': 6.5}
    
    rows = []
    for _ in range(n_samples):
        g = random.choices(genders, weights=[0.518, 0.482])[0]
        race = random.choices(race_ethnicity, weights=race_weights)[0]
        edu = random.choices(parental_education, weights=edu_weights)[0]
        lunch = random.choices(lunches, weights=lunch_weights)[0]
        prep = random.choices(test_preps, weights=prep_weights)[0]
        internet = random.choices(internet_opts, weights=internet_weights)[0]
        extra = random.choices(extra_opts, weights=extra_weights)[0]
        tutoring = random.choices(tutoring_opts, weights=tutoring_weights)[0]
        
        # Continuous behavioral and lifestyle features
        # Attendance: Beta distribution scaled to 55 - 100%
        att_raw = np.random.beta(7, 1.5)
        attendance = round(float(55 + att_raw * 45), 1)
        
        # Weekly Study Hours: Log-normal or Gamma distribution around 12-16 hrs
        study_hours = round(float(np.clip(np.random.gamma(4.5, 3.2), 2.0, 38.0)), 1)
        
        # Sleep Hours: Normal around 7.2 hrs (range 4 to 10)
        sleep_hours = round(float(np.clip(np.random.normal(7.2, 1.1), 4.0, 10.0)), 1)
        
        # Past Failures: Poisson around 0.35 (mostly 0, some 1, few 2, 3)
        past_fails = int(np.clip(np.random.poisson(0.35), 0, 4))
        
        # Base Academic Ability
        base = np.random.normal(67.5, 11.0)
        
        # Behavioral & Environmental Adjustments
        att_adj = (attendance - 85.0) * 0.35  # e.g. 95% -> +3.5 pts, 65% -> -7.0 pts
        study_adj = (study_hours - 12.0) * 0.42 # e.g. 20h -> +3.36 pts, 5h -> -2.94 pts
        fail_penalty = past_fails * -5.5       # Past backlog drag
        
        # Sleep curve (optimal between 7.0 and 8.5, fatigue penalty for < 6.0)
        if sleep_hours < 6.0:
            sleep_adj = (sleep_hours - 6.0) * 2.2
        elif sleep_hours > 9.0:
            sleep_adj = -1.0
        else:
            sleep_adj = 1.5
            
        env_adj = (
            edu_boost[edu] +
            lunch_boost[lunch] +
            prep_boost[prep] +
            internet_boost[internet] +
            tutoring_boost[tutoring] +
            att_adj +
            study_adj +
            fail_penalty +
            sleep_adj
        )
        
        g_math = 2.8 if g == 'male' else -2.8
        g_read = -2.8 if g == 'male' else 2.8
        g_write = -3.8 if g == 'male' else 3.8
        
        r = int(np.clip(round(base + env_adj + g_read + np.random.normal(0, 4.5)), 10, 100))
        w = int(np.clip(round((r * 0.65) + (base + env_adj + g_write) * 0.35 + np.random.normal(0, 3.0)), 10, 100))
        m = int(np.clip(round((r * 0.35) + (w * 0.25) + (base + env_adj + g_math) * 0.40 + np.random.normal(0, 3.8)), 10, 100))
        
        rows.append([
            g, race, edu, lunch, prep, internet, extra, tutoring,
            attendance, study_hours, sleep_hours, past_fails,
            r, w, m
        ])
        
    output_file = r"d:\My Project\student-performance-predictor\data\StudentsPerformance.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
            "internet_access",
            "extracurricular_activities",
            "tutoring_support",
            "attendance_rate",
            "weekly_study_hours",
            "sleep_hours_per_day",
            "past_failures",
            "reading score",
            "writing score",
            "math score"
        ])
        writer.writerows(rows)
        
    print(f"Successfully generated {len(rows)} records with 14 features in {output_file}")
    return output_file

if __name__ == "__main__":
    generate_full_dataset()
