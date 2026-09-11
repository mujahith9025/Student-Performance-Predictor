import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

def train_pass_fail_classifiers():
    print("=" * 75)
    print("   STUDENT PERFORMANCE PREDICTOR - PASS/FAIL & RISK CLASSIFICATION    ")
    print("=" * 75)
    sys.stdout.flush()

    # 1. Load Processed Feature Matrices
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train_continuous = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test_continuous = np.load(os.path.join(processed_dir, "y_test.npy"))

    # Convert continuous marks to binary classification (1: Pass >= 50, 0: At-Risk / Fail < 50)
    PASS_THRESHOLD = 50
    y_train = (y_train_continuous >= PASS_THRESHOLD).astype(int)
    y_test = (y_test_continuous >= PASS_THRESHOLD).astype(int)

    print(f"\n[1] Class Distribution (Pass Threshold = {PASS_THRESHOLD} marks):")
    print(f"    - Training Set: {np.sum(y_train == 1)} Pass, {np.sum(y_train == 0)} At-Risk ({np.mean(y_train)*100:.1f}% Pass rate)")
    print(f"    - Testing Set:  {np.sum(y_test == 1)} Pass, {np.sum(y_test == 0)} At-Risk ({np.mean(y_test)*100:.1f}% Pass rate)")
    sys.stdout.flush()

    # 2. Define Classifiers & Hyperparameter Tuning Grids
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
        },
        "Random Forest Classifier": {
            "model": RandomForestClassifier(random_state=42, n_jobs=1),
            "params": {"n_estimators": [50, 100], "max_depth": [4, 6, 8], "min_samples_split": [2, 5]}
        },
        "Gradient Boosting Classifier": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.03, 0.1], "max_depth": [3, 4]}
        },
        "Support Vector Classifier (SVC)": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
        },
        "K-Neighbors Classifier": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
        }
    }

    models_dir = os.path.join("artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)

    results = []
    trained_clf_dict = {}
    roc_curves = {}

    print("\n[2] Training & Evaluating Classification Algorithms (5-Fold Stratified CV):\n")
    print(f"{'Classifier':<30} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<9} | {'ROC-AUC':<8}")
    print("-" * 88)

    for name, config in classifiers.items():
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_
        trained_clf_dict[name] = best_clf

        # Test Set Predictions
        y_pred = best_clf.predict(X_test)
        y_prob = best_clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

        # Save individual classifier
        safe_fname = "clf_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(best_clf, os.path.join(models_dir, safe_fname))

        results.append({
            "Classifier": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Best Params": str(grid.best_params_),
            "Filename": safe_fname
        })

        print(f"{name:<30} | {acc*100:<8.2f}% | {prec:<10.4f} | {rec:<8.4f} | {f1:<9.4f} | {auc:<8.4f}")
        sys.stdout.flush()

    # 3. Compile Leaderboard & Select Best Classifier
    clf_df = pd.DataFrame(results)
    clf_df = clf_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    metrics_csv = os.path.join("artifacts", "classifier_metrics.csv")
    clf_df.to_csv(metrics_csv, index=False)
    print(f"\n[3] Saved Classifier Metrics to: {metrics_csv}")

    best_clf_info = clf_df.iloc[0]
    best_clf_name = best_clf_info["Classifier"]
    best_clf_obj = trained_clf_dict[best_clf_name]
    best_clf_dest = os.path.join("artifacts", "best_classifier.joblib")
    joblib.dump(best_clf_obj, best_clf_dest)

    print(f"\n[4] Champion Risk Classifier: ** {best_clf_name} **")
    print(f"    - Accuracy: {best_clf_info['Accuracy']*100:.2f}%")
    print(f"    - ROC-AUC:  {best_clf_info['ROC-AUC']:.4f}")
    print(f"    - Saved to: {best_clf_dest}")

    # 4. Visualizations: Confusion Matrix & ROC Curves
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # A. Confusion Matrix for Best Classifier
    plt.figure(figsize=(6, 5))
    best_y_pred = best_clf_obj.predict(X_test)
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["At-Risk / Fail (0)", "Pass (1)"],
        yticklabels=["At-Risk / Fail (0)", "Pass (1)"],
        cbar=False
    )
    plt.title(f"Confusion Matrix: {best_clf_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "08_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[5] Generated Visuals:")
    print(f"    - Saved Confusion Matrix: {cm_path}")

    # B. ROC-AUC Curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance (AUC = 0.500)")
    plt.title("ROC Curves: Academic Pass / Fail Classification", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "09_roc_auc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"    - Saved ROC-AUC Curves:   {roc_path}")

    print("\n[OK] Classification & Risk Prediction Pipeline Completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    train_pass_fail_classifiers()
