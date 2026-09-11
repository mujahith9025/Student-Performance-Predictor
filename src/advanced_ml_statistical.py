import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.advanced_feature_engineering import engineer_features

ARCHETYPE_LABELS = {
    0: {
        "name": "🌟 Honors & Self-Directed Scholars",
        "tag": "Honors / Distinction",
        "badge_color": "#059669",
        "bg_color": "#ECFDF5",
        "summary": "High attendance (>92%), consistent study hours (18-30h/wk), low backlogs, and balanced sleep. Excels across all subjects."
    },
    1: {
        "name": "⚡ High-Effort Dedicated Strivers",
        "tag": "High Effort / Math Anxious",
        "badge_color": "#2563EB",
        "bg_color": "#EFF6FF",
        "summary": "High study volume (>15h/wk) and attendance, but seeks concept reinforcement in quantitative problem solving."
    },
    2: {
        "name": "📖 Linguistic-Dominant Explorers",
        "tag": "Verbal Strong / Needs Math Prep",
        "badge_color": "#7C3AED",
        "bg_color": "#F5F3FF",
        "summary": "Strong reading and writing literacy, moderate study hours, benefits significantly from structured test preparation."
    },
    3: {
        "name": "🚨 Foundational Support & At-Risk",
        "tag": "Urgent Remedial Priority",
        "badge_color": "#DC2626",
        "bg_color": "#FEF2F2",
        "summary": "Attendance friction (<80%), prior backlogs, lower self-study time. Requires structured peer tutoring and wellness monitoring."
    }
}

def train_advanced_statistical_suite():
    """
    Trains and saves:
    1. Multi-Output 3-Subject Regressor (Math, Reading, Writing)
    2. Conformal Prediction Uncertainty Quantifier (95% Confidence Intervals)
    3. Unsupervised Behavioral Archetype Clusterer (K-Means & 2D/3D PCA)
    """
    print("=" * 80)
    print("   ADVANCED MACHINE LEARNING & STATISTICAL ENHANCEMENT ENGINE TRAINING   ")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Load Data & Preprocessor
    data_path = os.path.join("data", "StudentsPerformance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df_raw = pd.read_csv(data_path)

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))

    # Feature Engineering
    df_eng = engineer_features(df_raw)
    X_transformed = preprocessor.transform(df_eng)
    
    # Target Variables
    y_math = df_raw["math score"].values
    y_reading = df_raw["reading score"].values
    y_writing = df_raw["writing score"].values
    y_multi = np.column_stack([y_math, y_reading, y_writing])

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_multi, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 1. MULTI-OUTPUT 3-SUBJECT REGRESSOR
    # ---------------------------------------------------------
    print("\n[1/3] Training Multi-Output 3-Subject Regression Engine (Math, Reading, Writing)...")
    base_reg = Ridge(alpha=1.0)
    multi_model = MultiOutputRegressor(base_reg)
    multi_model.fit(X_train, y_train)

    multi_preds = multi_model.predict(X_test)
    r2_math = r2_score(y_test[:, 0], multi_preds[:, 0])
    r2_read = r2_score(y_test[:, 1], multi_preds[:, 1])
    r2_write = r2_score(y_test[:, 2], multi_preds[:, 2])
    mae_math = mean_absolute_error(y_test[:, 0], multi_preds[:, 0])

    print(f"      - Math Forecast R²:    {r2_math * 100:.2f}% (MAE: ±{mae_math:.2f} marks)")
    print(f"      - Reading Forecast R²: {r2_read * 100:.2f}%")
    print(f"      - Writing Forecast R²: {r2_write * 100:.2f}%")
    print(f"      - Multi-Subject Joint R²: {np.mean([r2_math, r2_read, r2_write]) * 100:.2f}%")

    joblib.dump(multi_model, os.path.join(artifacts_dir, "multi_subject_model.joblib"))
    
    multi_metrics_df = pd.DataFrame([
        {"Subject": "Mathematics", "R2": round(r2_math, 4), "MAE": round(mae_math, 2)},
        {"Subject": "Reading", "R2": round(r2_read, 4), "MAE": round(mean_absolute_error(y_test[:, 1], multi_preds[:, 1]), 2)},
        {"Subject": "Writing", "R2": round(r2_write, 4), "MAE": round(mean_absolute_error(y_test[:, 2], multi_preds[:, 2]), 2)}
    ])
    multi_metrics_df.to_csv(os.path.join(artifacts_dir, "multi_subject_metrics.csv"), index=False)

    # ---------------------------------------------------------
    # 2. CONFORMAL PREDICTION UNCERTAINTY QUANTIFIER
    # ---------------------------------------------------------
    print("\n[2/3] Calibrating Conformal Prediction Non-Conformity Residual Distribution...")
    math_residuals = np.abs(y_test[:, 0] - multi_preds[:, 0])
    
    # Compute 80%, 90%, 95%, and 99% Conformal Error Quantiles
    q80 = float(np.quantile(math_residuals, 0.80))
    q90 = float(np.quantile(math_residuals, 0.90))
    q95 = float(np.quantile(math_residuals, 0.95))
    q99 = float(np.quantile(math_residuals, 0.99))

    uncertainty_calibrator = {
        "residual_mean": float(np.mean(math_residuals)),
        "residual_std": float(np.std(math_residuals)),
        "q80_margin": round(q80, 2),
        "q90_margin": round(q90, 2),
        "q95_margin": round(q95, 2),
        "q99_margin": round(q99, 2),
        "sample_size": len(math_residuals)
    }

    print(f"      - 90% Confidence Interval Margin: ±{q90:.2f} marks")
    print(f"      - 95% Confidence Interval Margin: ±{q95:.2f} marks")
    print(f"      - 99% Extreme Bound Margin:       ±{q99:.2f} marks")

    joblib.dump(uncertainty_calibrator, os.path.join(artifacts_dir, "uncertainty_model.joblib"))

    # ---------------------------------------------------------
    # 3. UNSUPERVISED BEHAVIORAL ARCHETYPE DISCOVERY (K-MEANS & PCA)
    # ---------------------------------------------------------
    print("\n[3/3] Training K-Means (k=4) Behavioral Clusterer & 2D/3D PCA Decomposer...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_transformed)

    pca_2d = PCA(n_components=2, random_state=42)
    pca_coords_2d = pca_2d.fit_transform(X_transformed)
    
    explained_var = pca_2d.explained_variance_ratio_
    print(f"      - 2D PCA Total Explained Variance: {np.sum(explained_var) * 100:.2f}% (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%)")

    # Map Clusters to Archetypes based on Math Score means
    cluster_means = [df_raw.loc[cluster_labels == i, "math score"].mean() for i in range(4)]
    sorted_cluster_order = np.argsort(cluster_means)[::-1] # 0 = highest, 3 = lowest
    cluster_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(sorted_cluster_order)}

    archetype_labels_mapped = np.array([cluster_mapping[c] for c in cluster_labels])
    df_raw["Archetype_ID"] = archetype_labels_mapped
    df_raw["Archetype_Name"] = [ARCHETYPE_LABELS[a]["name"] for a in archetype_labels_mapped]
    df_raw["PCA_1"] = pca_coords_2d[:, 0]
    df_raw["PCA_2"] = pca_coords_2d[:, 1]

    # Save Cluster Summary
    archetype_summary = []
    for a_id in range(4):
        sub_df = df_raw[df_raw["Archetype_ID"] == a_id]
        archetype_summary.append({
            "Archetype ID": a_id,
            "Name": ARCHETYPE_LABELS[a_id]["name"],
            "Student Count": len(sub_df),
            "Percent of Cohort": f"{len(sub_df) / len(df_raw) * 100:.1f}%",
            "Avg Math Score": round(sub_df["math score"].mean(), 1),
            "Avg Attendance": round(sub_df["attendance_rate"].mean(), 1),
            "Avg Study Hours": round(sub_df["weekly_study_hours"].mean(), 1),
            "Avg Sleep Hours": round(sub_df["sleep_hours_per_day"].mean(), 1),
            "Avg Failures": round(sub_df["past_failures"].mean(), 2)
        })

    arch_df = pd.DataFrame(archetype_summary)
    arch_df.to_csv(os.path.join(artifacts_dir, "archetype_summary.csv"), index=False)

    cluster_bundle = {
        "kmeans": kmeans,
        "cluster_mapping": cluster_mapping,
        "pca_2d": pca_2d,
        "cluster_labels": archetype_labels_mapped,
        "pca_coords": pca_coords_2d,
        "explained_variance": explained_var,
        "reference_df": df_raw[["Archetype_ID", "Archetype_Name", "PCA_1", "PCA_2", "math score", "reading score", "attendance_rate", "weekly_study_hours"]]
    }
    joblib.dump(cluster_bundle, os.path.join(artifacts_dir, "archetype_clusterer.joblib"))

    print("\n" + "=" * 80)
    print(" ALL ADVANCED STATISTICAL ARTIFACTS GENERATED AND PERSISTED SUCCESSFULLY! ")
    print("=" * 80)
    return True

def predict_with_confidence_intervals(transformed_input, model, uncertainty_dict):
    """
    Computes exact point prediction plus 90% and 95% non-parametric Conformal Prediction Intervals.
    """
    pred_point = float(np.clip(model.predict(transformed_input)[0], 0, 100))
    margin_90 = uncertainty_dict.get("q90_margin", 3.1)
    margin_95 = uncertainty_dict.get("q95_margin", 3.47)
    
    ci_90_lower = max(0.0, round(pred_point - margin_90, 1))
    ci_90_upper = min(100.0, round(pred_point + margin_90, 1))
    
    ci_95_lower = max(0.0, round(pred_point - margin_95, 1))
    ci_95_upper = min(100.0, round(pred_point + margin_95, 1))
    
    return {
        "predicted_score": pred_point,
        "margin_90": margin_90,
        "margin_95": margin_95,
        "ci_90_range": (ci_90_lower, ci_90_upper),
        "ci_95_range": (ci_95_lower, ci_95_upper),
        "ci_95_str": f"{pred_point:.1f} ± {margin_95:.1f} marks [{ci_95_lower} – {ci_95_upper}]"
    }

def classify_student_archetype(transformed_input, cluster_bundle):
    """
    Determines student behavioral archetype, cluster distances, and 2D PCA location coordinates.
    """
    kmeans = cluster_bundle["kmeans"]
    cluster_mapping = cluster_bundle["cluster_mapping"]
    pca_2d = cluster_bundle["pca_2d"]
    
    raw_cluster = kmeans.predict(transformed_input)[0]
    archetype_id = cluster_mapping.get(raw_cluster, 0)
    
    pca_pt = pca_2d.transform(transformed_input)[0]
    
    # Calculate Soft Distance Distribution
    distances = kmeans.transform(transformed_input)[0]
    inv_dist = 1.0 / (distances + 1e-5)
    probs = inv_dist / np.sum(inv_dist)
    
    info = ARCHETYPE_LABELS[archetype_id]
    
    return {
        "archetype_id": archetype_id,
        "name": info["name"],
        "tag": info["tag"],
        "badge_color": info["badge_color"],
        "bg_color": info["bg_color"],
        "summary": info["summary"],
        "pca_x": float(pca_pt[0]),
        "pca_y": float(pca_pt[1]),
        "affinity_score": float(np.max(probs) * 100.0)
    }

if __name__ == "__main__":
    train_advanced_statistical_suite()
