import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 11

def perform_eda():
    data_path = os.path.join("data", "StudentsPerformance.csv")
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 60)
    print("           STUDENT PERFORMANCE PREDICTOR - EDA           ")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    # 2. Check Data Types and Missing Values
    print("\n[2] Data Types & Missing Values Check:")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Null Percentage (%)': (df.isnull().sum() / len(df)) * 100
    })
    print(info_df)
    
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicates}")
    
    # 3. Summary Statistics for Numerical Columns
    print("\n[3] Numerical Summary Statistics:")
    print(df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    
    # 4. Summary Statistics for Categorical Columns
    print("\n[4] Categorical Columns Summary:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nDistribution of '{col}':")
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages.round(2)})
        print(dist_df)
        
    # 5. Grouped Insights
    print("\n[5] Average Scores by Key Categories:")
    print("\n--- Impact of Test Preparation Course ---")
    print(df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean().round(2))
    
    print("\n--- Impact of Lunch Type ---")
    print(df.groupby('lunch')[['math score', 'reading score', 'writing score']].mean().round(2))
    
    print("\n--- Impact of Parental Education ---")
    print(df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean().round(2))

    # ==================== VISUALIZATIONS ====================
    print("\n[6] Generating EDA Visualizations...")
    
    # 1. Distribution of Scores
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scores = ['math score', 'reading score', 'writing score']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, score, color in zip(axes, scores, colors):
        sns.histplot(df[score], kde=True, ax=ax, color=color, bins=20)
        mean_val = df[score].mean()
        median_val = df[score].median()
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='black', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.1f}')
        ax.set_title(f'Distribution of {score.title()}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Score (0-100)')
        ax.set_ylabel('Student Count')
        ax.legend()
        
    plt.tight_layout()
    dist_path = os.path.join(plots_dir, "01_score_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f" Saved: {dist_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    corr = df[['math score', 'reading score', 'writing score']].corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', vmin=0, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Correlation Matrix Between Examination Scores', fontsize=13, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(plots_dir, "02_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f" Saved: {corr_path}")

    # 3. Test Preparation Impact
    plt.figure(figsize=(8, 5))
    prep_melted = pd.melt(df, id_vars=['test preparation course'], 
                          value_vars=['math score', 'reading score', 'writing score'],
                          var_name='Subject', value_name='Score')
    prep_melted['Subject'] = prep_melted['Subject'].str.replace(' score', '').str.capitalize()
    sns.barplot(data=prep_melted, x='Subject', y='Score', hue='test preparation course', palette='Set2')
    plt.title('Impact of Test Preparation Course on Student Scores', fontsize=13, fontweight='bold')
    plt.xlabel('Exam Subject')
    plt.ylabel('Average Score')
    plt.ylim(0, 100)
    plt.legend(title='Test Prep Course')
    plt.tight_layout()
    prep_path = os.path.join(plots_dir, "03_test_prep_impact.png")
    plt.savefig(prep_path, dpi=300)
    plt.close()
    print(f" Saved: {prep_path}")

    # 4. Parental Level of Education Impact
    plt.figure(figsize=(10, 5))
    edu_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    sns.boxplot(data=df, x='parental level of education', y='math score', order=edu_order, palette='Spectral')
    plt.title('Math Score Distribution by Parental Level of Education', fontsize=13, fontweight='bold')
    plt.xlabel('Parental Level of Education')
    plt.ylabel('Math Score')
    plt.xticks(rotation=20)
    plt.tight_layout()
    edu_path = os.path.join(plots_dir, "04_parental_education_impact.png")
    plt.savefig(edu_path, dpi=300)
    plt.close()
    print(f" Saved: {edu_path}")

    # 5. Lunch Type Impact
    plt.figure(figsize=(8, 5))
    lunch_melted = pd.melt(df, id_vars=['lunch'], 
                           value_vars=['math score', 'reading score', 'writing score'],
                           var_name='Subject', value_name='Score')
    lunch_melted['Subject'] = lunch_melted['Subject'].str.replace(' score', '').str.capitalize()
    sns.barplot(data=lunch_melted, x='Subject', y='Score', hue='lunch', palette='coolwarm')
    plt.title('Score Comparison by Lunch Type (Standard vs. Free/Reduced)', fontsize=13, fontweight='bold')
    plt.xlabel('Exam Subject')
    plt.ylabel('Average Score')
    plt.ylim(0, 100)
    plt.legend(title='Lunch Type')
    plt.tight_layout()
    lunch_path = os.path.join(plots_dir, "05_lunch_impact.png")
    plt.savefig(lunch_path, dpi=300)
    plt.close()
    print(f" Saved: {lunch_path}")

    print("\n[OK] EDA Complete! All charts successfully generated.")

if __name__ == "__main__":
    perform_eda()
