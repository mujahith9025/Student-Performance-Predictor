import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_models():
    print("=" * 70)
    print("       STUDENT PERFORMANCE PREDICTOR - MODEL EVALUATION       ")
    print("=" * 70)
    sys.stdout.flush()

    # 1. Load Processed Data
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"\n[1] Loaded Test Set: {X_test.shape[0]} unseen student records")

    # 2. Load Saved Models
    models_dir = os.path.join("artifacts", "models")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]

    results = []
    predictions_dict = {}

    print("\n[2] Evaluating Models on Unseen Test Data:\n")
    print(f"{'Model Name':<28} | {'Train R2':<9} | {'Test R2':<9} | {'Test MAE':<9} | {'Test RMSE':<9}")
    print("-" * 75)

    for file_name in sorted(model_files):
        model_path = os.path.join(models_dir, file_name)
        model = joblib.load(model_path)
        
        display_name = file_name.replace(".joblib", "").replace("_", " ").title()

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predictions_dict[display_name] = y_test_pred

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        results.append({
            "Model": display_name,
            "Train R2 Score": round(train_r2, 4),
            "Test R2 Score": round(test_r2, 4),
            "Test MAE (marks)": round(mae, 2),
            "Test RMSE (marks)": round(rmse, 2),
            "Filename": file_name
        })

        print(f"{display_name:<28} | {train_r2:<9.4f} | {test_r2:<9.4f} | {mae:<9.2f} | {rmse:<9.2f}")
        sys.stdout.flush()

    # 3. Compile Comparison DataFrame & Rank
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values(by="Test R2 Score", ascending=False).reset_index(drop=True)
    
    # Save Metrics Table
    metrics_path = os.path.join("artifacts", "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[3] Saved Model Comparison Metrics to: {metrics_path}")

    # 4. Identify Champion (Best) Model
    best_model_info = metrics_df.iloc[0]
    best_model_name = best_model_info["Model"]
    best_model_file = best_model_info["Filename"]
    best_model = joblib.load(os.path.join(models_dir, best_model_file))
    
    best_model_dest = os.path.join("artifacts", "best_model.joblib")
    joblib.dump(best_model, best_model_dest)
    print(f"\n[4] Champion Model Selected: ** {best_model_name} **")
    print(f"    - Test R2 Score: {best_model_info['Test R2 Score'] * 100:.2f}%")
    print(f"    - Average Prediction Error (MAE): +/- {best_model_info['Test MAE (marks)']} marks")
    print(f"    - Saved Best Model to: {best_model_dest}")

    # 5. Visualizations
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Chart A: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="Model", y="Test R2 Score", palette="viridis")
    plt.title("Model Comparison: Test R² Score on Unseen Data", fontsize=13, fontweight="bold")
    plt.ylabel("R² Score (Higher is Better)")
    plt.xlabel("Machine Learning Algorithm")
    plt.ylim(0, 1.05)
    for i, row in metrics_df.iterrows():
        plt.text(i, row["Test R2 Score"] + 0.02, f"{row['Test R2 Score']:.3f}", ha="center", fontweight="bold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    comp_plot_path = os.path.join(plots_dir, "06_model_performance_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"\n[5] Generated Evaluation Charts:")
    print(f"    - Saved: {comp_plot_path}")

    # Chart B: Actual vs Predicted Scatter for Champion Model
    plt.figure(figsize=(7, 6))
    best_preds = predictions_dict[best_model_name]
    plt.scatter(y_test, best_preds, alpha=0.6, color="#2980b9", edgecolors="w", s=60)
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label="Perfect 1:1 Prediction Line")
    plt.title(f"Actual vs. Predicted Math Scores ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Math Score")
    plt.ylabel("Predicted Math Score")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    scatter_plot_path = os.path.join(plots_dir, "07_actual_vs_predicted.png")
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"    - Saved: {scatter_plot_path}")

    print("\n[OK] Phase 5 Model Evaluation & Selection Completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    evaluate_models()
