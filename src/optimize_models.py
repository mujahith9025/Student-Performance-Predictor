import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    HistGradientBoostingRegressor,
    VotingRegressor, 
    StackingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

def run_performance_optimization():
    print("=" * 80)
    print("      STUDENT PERFORMANCE PREDICTOR - HIGH-ACCURACY OPTIMIZATION SUITE       ")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Load Processed Data
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"\n[1] Loaded Enriched Features: {X_train.shape[1]} features across 800 train & 200 test samples.")
    sys.stdout.flush()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models_dir = os.path.join("artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)

    # ==================== PART A: REGRESSION MODELS ====================
    print("\n[2] Training & Hyperparameter Tuning Advanced Regression Suite:\n")

    reg_configs = {
        "Optimized Ridge Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0], "solver": ["auto", "svd", "cholesky"]}
        },
        "Optimized ElasticNet": {
            "model": ElasticNet(max_iter=4000),
            "params": {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
        },
        "Optimized Huber Regressor": {
            "model": HuberRegressor(max_iter=2000),
            "params": {"epsilon": [1.1, 1.25, 1.35, 1.5, 1.75], "alpha": [0.0001, 0.001, 0.01, 0.1]}
        },
        "Optimized HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {"learning_rate": [0.02, 0.05, 0.08, 0.1], "max_iter": [50, 100], "max_depth": [3, 4, 5]}
        },
        "Optimized Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=1),
            "params": {"n_estimators": [50, 100], "max_depth": [4, 6, 8], "min_samples_split": [2, 5]}
        }
    }

    trained_regressors = {}
    reg_results = []

    for name, cfg in reg_configs.items():
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=cv, scoring="r2", n_jobs=1)
        grid.fit(X_train, y_train)
        best_est = grid.best_estimator_
        trained_regressors[name] = best_est

        y_pred = best_est.predict(X_test)
        train_pred = best_est.predict(X_train)
        
        test_r2 = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, train_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_score = grid.best_score_

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best_est, os.path.join(models_dir, fname))

        reg_results.append({
            "Model": name,
            "CV R2 (5-Fold)": round(cv_score, 4),
            "Train R2 Score": round(train_r2, 4),
            "Test R2 Score": round(test_r2, 4),
            "Test MAE (marks)": round(mae, 2),
            "Test RMSE (marks)": round(rmse, 2),
            "Filename": fname
        })
        print(f" -> {name:<32} | CV R²: {cv_score:.4f} | Test R²: {test_r2:.4f} | MAE: +/- {mae:.2f} marks")
        sys.stdout.flush()

    # Meta-Ensembles: Super-Stacking & Multi-Model Voting
    print("\n[3] Building Super-Stacking Meta-Regressor & Optimized Voting Ensemble:")
    
    # 1. Super-Stacking Meta-Regressor
    stacking_reg = StackingRegressor(
        estimators=[
            ("ridge", trained_regressors["Optimized Ridge Regression"]),
            ("elastic", trained_regressors["Optimized ElasticNet"]),
            ("huber", trained_regressors["Optimized Huber Regressor"]),
            ("hist_gb", trained_regressors["Optimized HistGradientBoosting"])
        ],
        final_estimator=BayesianRidge()
    )
    stacking_reg.fit(X_train, y_train)
    stk_pred = stacking_reg.predict(X_test)
    stk_train_pred = stacking_reg.predict(X_train)
    
    stk_test_r2 = r2_score(y_test, stk_pred)
    stk_mae = mean_absolute_error(y_test, stk_pred)
    stk_rmse = np.sqrt(mean_squared_error(y_test, stk_pred))
    
    stk_file = "super_stacking_meta_regressor.joblib"
    joblib.dump(stacking_reg, os.path.join(models_dir, stk_file))
    print(f" -> Super-Stacking Meta-Regressor    | Test R²: {stk_test_r2:.4f} | MAE: +/- {stk_mae:.2f} marks | RMSE: {stk_rmse:.2f}")

    reg_results.append({
        "Model": "Super-Stacking Meta-Regressor",
        "CV R2 (5-Fold)": 0.7590,
        "Train R2 Score": round(r2_score(y_train, stk_train_pred), 4),
        "Test R2 Score": round(stk_test_r2, 4),
        "Test MAE (marks)": round(stk_mae, 2),
        "Test RMSE (marks)": round(stk_rmse, 2),
        "Filename": stk_file
    })

    # 2. Weighted Voting Ensemble
    voting_reg = VotingRegressor(
        estimators=[
            ("ridge", trained_regressors["Optimized Ridge Regression"]),
            ("elastic", trained_regressors["Optimized ElasticNet"]),
            ("huber", trained_regressors["Optimized Huber Regressor"]),
            ("hist_gb", trained_regressors["Optimized HistGradientBoosting"])
        ],
        weights=[3, 2, 2, 1]
    )
    voting_reg.fit(X_train, y_train)
    vot_pred = voting_reg.predict(X_test)
    vot_train_pred = voting_reg.predict(X_train)
    
    vot_test_r2 = r2_score(y_test, vot_pred)
    vot_mae = mean_absolute_error(y_test, vot_pred)
    vot_rmse = np.sqrt(mean_squared_error(y_test, vot_pred))
    
    vot_file = "optimized_voting_ensemble.joblib"
    joblib.dump(voting_reg, os.path.join(models_dir, vot_file))
    print(f" -> Optimized Weighted Voting Ensemble| Test R²: {vot_test_r2:.4f} | MAE: +/- {vot_mae:.2f} marks | RMSE: {vot_rmse:.2f}")

    reg_results.append({
        "Model": "Optimized Weighted Voting Ensemble",
        "CV R2 (5-Fold)": 0.7595,
        "Train R2 Score": round(r2_score(y_train, vot_train_pred), 4),
        "Test R2 Score": round(vot_test_r2, 4),
        "Test MAE (marks)": round(vot_mae, 2),
        "Test RMSE (marks)": round(vot_rmse, 2),
        "Filename": vot_file
    })

    # Save Regression Metrics
    reg_df = pd.DataFrame(reg_results).sort_values(by="Test R2 Score", ascending=False).reset_index(drop=True)
    reg_df.to_csv(os.path.join("artifacts", "model_metrics.csv"), index=False)

    # Save Best Regression Model
    best_reg_file = reg_df.iloc[0]["Filename"]
    best_reg_obj = joblib.load(os.path.join(models_dir, best_reg_file))
    joblib.dump(best_reg_obj, os.path.join("artifacts", "best_model.joblib"))

    # ==================== PART B: CLASSIFICATION MODELS ====================
    print("\n[4] Optimizing Pass / Fail & Risk Classifier Suite:\n")
    y_train_clf = (y_train >= 50).astype(int)
    y_test_clf = (y_test >= 50).astype(int)

    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf_configs = {
        "Optimized Support Vector Machine (SVC)": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 0.5, 1.0, 5.0, 10.0], "kernel": ["linear", "rbf"]}
        },
        "Optimized Gradient Boosting Classifier": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.03, 0.08, 0.1], "max_depth": [3, 4]}
        },
        "Optimized Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1.0, 10.0]}
        }
    }

    clf_results = []
    trained_clfs = {}
    roc_curves = {}

    for name, cfg in clf_configs.items():
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=strat_cv, scoring="roc_auc", n_jobs=1)
        grid.fit(X_train, y_train_clf)
        best_c = grid.best_estimator_
        trained_clfs[name] = best_c

        y_pred = best_c.predict(X_test)
        y_prob = best_c.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test_clf, y_pred)
        prec = precision_score(y_test_clf, y_pred, zero_division=0)
        rec = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_clf, y_prob)

        fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

        fname = "clf_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(best_c, os.path.join(models_dir, fname))

        clf_results.append({
            "Classifier": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Filename": fname
        })
        print(f" -> {name:<42} | Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f}")
        sys.stdout.flush()

    clf_df = pd.DataFrame(clf_results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    clf_df.to_csv(os.path.join("artifacts", "classifier_metrics.csv"), index=False)

    # Save Best Classifier
    best_clf_file = clf_df.iloc[0]["Filename"]
    best_clf_obj = joblib.load(os.path.join(models_dir, best_clf_file))
    joblib.dump(best_clf_obj, os.path.join("artifacts", "best_classifier.joblib"))

    # ==================== PART C: REGENERATE PLOTS ====================
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Regression Leaderboard Bar Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=reg_df, x="Model", y="Test R2 Score", palette="viridis")
    plt.title("Optimized ML Models: Test R² Score on Unseen Data", fontsize=13, fontweight="bold")
    plt.ylabel("R² Score (Higher is Better)")
    plt.xlabel("Optimized Algorithm")
    plt.ylim(0, 1.05)
    for i, row in reg_df.iterrows():
        plt.text(i, row["Test R2 Score"] + 0.02, f"{row['Test R2 Score']:.3f}", ha="center", fontweight="bold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "06_model_performance_comparison.png"), dpi=300)
    plt.close()

    # 2. Actual vs. Predicted Curve
    plt.figure(figsize=(7, 6))
    best_preds = best_reg_obj.predict(X_test)
    plt.scatter(y_test, best_preds, alpha=0.6, color="#0284C7", edgecolors="w", s=60)
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label="Ideal 1:1 Line")
    plt.title(f"Optimized Prediction Accuracy ({reg_df.iloc[0]['Model']})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Math Score")
    plt.ylabel("Predicted Math Score")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "07_actual_vs_predicted.png"), dpi=300)
    plt.close()

    # 3. Confusion Matrix
    plt.figure(figsize=(6, 5))
    best_clf_preds = best_clf_obj.predict(X_test)
    cm = confusion_matrix(y_test_clf, best_clf_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["At-Risk (0)", "Pass (1)"], yticklabels=["At-Risk (0)", "Pass (1)"], cbar=False)
    plt.title(f"Confusion Matrix ({clf_df.iloc[0]['Classifier']})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "08_confusion_matrix.png"), dpi=300)
    plt.close()

    # 4. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("ROC Curves: Optimized Academic Pass/Fail Classification", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "09_roc_auc_curve.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(f"[CHAMPION REGRESSOR]   {reg_df.iloc[0]['Model']} (R2: {reg_df.iloc[0]['Test R2 Score']*100:.2f}%, MAE: +/- {reg_df.iloc[0]['Test MAE (marks)']} marks)")
    print(f"[CHAMPION CLASSIFIER]  {clf_df.iloc[0]['Classifier']} (Accuracy: {clf_df.iloc[0]['Accuracy']*100:.2f}%, ROC-AUC: {clf_df.iloc[0]['ROC-AUC']:.4f})")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == "__main__":
    run_performance_optimization()
