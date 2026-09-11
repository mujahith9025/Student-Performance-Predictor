import csv
import random

random.seed(42)

genders = ['female', 'male']
race_ethnicity = ['group A', 'group B', 'group C', 'group D', 'group E']
race_weights = [0.089, 0.190, 0.319, 0.262, 0.140]

parental_education = ["some college", "associate's degree", "high school", "some high school", "bachelor's degree", "master's degree"]
edu_weights = [0.226, 0.222, 0.196, 0.179, 0.118, 0.059]

lunches = ['standard', 'free/reduced']
lunch_weights = [0.645, 0.355]

test_preps = ['none', 'completed']
prep_weights = [0.642, 0.358]

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

rows = []
for _ in range(1000):
    g = random.choices(genders, weights=[0.518, 0.482])[0]
    race = random.choices(race_ethnicity, weights=race_weights)[0]
    edu = random.choices(parental_education, weights=edu_weights)[0]
    lunch = random.choices(lunches, weights=lunch_weights)[0]
    prep = random.choices(test_preps, weights=prep_weights)[0]
    
    base = random.gauss(68, 12)
    adj = edu_boost[edu] + lunch_boost[lunch] + prep_boost[prep]
    
    g_math = 3 if g == 'male' else -3
    g_read = -3 if g == 'male' else 3
    g_write = -4 if g == 'male' else 4
    
    m = int(max(0, min(100, round(base + adj + g_math + random.gauss(0, 6)))))
    r = int(max(0, min(100, round(base + adj + g_read + random.gauss(0, 5)))))
    w = int(max(0, min(100, round((r * 0.7) + (base + adj + g_write) * 0.3 + random.gauss(0, 3)))))
    
    rows.append([g, race, edu, lunch, prep, m, r, w])

output_file = r"d:\My Project\student-performance-predictor\data\StudentsPerformance.csv"
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ])
    writer.writerows(rows)

print(f"Successfully generated {len(rows)} records in {output_file}")
