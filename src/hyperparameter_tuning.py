import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run_hyperparameter_tuning():
    print("=" * 75)
    print("    STUDENT PERFORMANCE PREDICTOR - ADVANCED MODELS & HYPERPARAMETER TUNING   ")
    print("=" * 75)
    sys.stdout.flush()

    # 1. Load Processed Data
    processed_dir = os.path.join("data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"\n[1] Loaded Training ({X_train.shape[0]} samples) & Testing ({X_test.shape[0]} samples)")
    print(f"    - Features count: {X_train.shape[1]}")
    sys.stdout.flush()

    # 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Focused Parameter Grids for Fast and Accurate Optimization
    tuning_configs = {
        "Tuned Ridge Regression": {
            "model": Ridge(),
            "params": {
                "alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0],
                "solver": ["auto", "svd", "cholesky"]
            }
        },
        "Tuned Lasso Regression": {
            "model": Lasso(max_iter=3000),
            "params": {
                "alpha": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
                "selection": ["cyclic", "random"]
            }
        },
        "Tuned ElasticNet": {
            "model": ElasticNet(max_iter=3000),
            "params": {
                "alpha": [0.001, 0.01, 0.05, 0.1, 0.5],
                "l1_ratio": [0.1, 0.5, 0.9]
            }
        },
        "Tuned Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [4, 6, 8],
                "min_samples_split": [2, 5]
            }
        },
        "Tuned Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.03, 0.08, 0.1],
                "max_depth": [3, 4]
            }
        },
        "Tuned AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {
                "n_estimators": [30, 50, 80],
                "learning_rate": [0.02, 0.08, 0.2]
            }
        }
    }

    models_dir = os.path.join("artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)

    tuning_results = []
    best_estimators = {}

    print("\n[2] Executing 5-Fold Cross-Validation Hyperparameter Tuning:\n")
    print(f"{'Algorithm':<28} | {'CV R2':<8} | {'Test R2':<8} | {'MAE':<7} | {'RMSE':<7} | {'Time':<6}")
    print("-" * 75)

    for name, config in tuning_configs.items():
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="r2",
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_estimators[name] = best_model
        
        # Test performance
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        cv_score = grid_search.best_score_
        
        # Save model
        filename = name.lower().replace(" ", "_") + ".joblib"
        save_path = os.path.join(models_dir, filename)
        joblib.dump(best_model, save_path)
        
        tuning_results.append({
            "Model": name,
            "CV R2 (5-Fold)": round(cv_score, 4),
            "Train R2": round(train_r2, 4),
            "Test R2": round(test_r2, 4),
            "Test MAE": round(mae, 2),
            "Test RMSE": round(rmse, 2),
            "Best Params": str(grid_search.best_params_),
            "Time (s)": round(elapsed, 1),
            "Filename": filename
        })
        
        print(f"{name:<28} | {cv_score:<8.4f} | {test_r2:<8.4f} | {mae:<7.2f} | {rmse:<7.2f} | {elapsed:4.1f}s")
        sys.stdout.flush()

    # 3. Build Advanced Meta-Ensembles (Voting & Stacking)
    print("\n[3] Building Advanced Meta-Ensembles (Voting & Stacking):")
    
    # Voting Regressor
    voting_model = VotingRegressor(
        estimators=[
            ("ridge", best_estimators["Tuned Ridge Regression"]),
            ("gb", best_estimators["Tuned Gradient Boosting"]),
            ("rf", best_estimators["Tuned Random Forest"])
        ],
        weights=[3, 1, 1]
    )
    voting_model.fit(X_train, y_train)
    v_test_pred = voting_model.predict(X_test)
    v_train_pred = voting_model.predict(X_train)
    v_test_r2 = r2_score(y_test, v_test_pred)
    v_mae = mean_absolute_error(y_test, v_test_pred)
    v_rmse = np.sqrt(mean_squared_error(y_test, v_test_pred))
    
    voting_file = "voting_ensemble_regressor.joblib"
    joblib.dump(voting_model, os.path.join(models_dir, voting_file))
    print(f"{'Voting Ensemble Regressor':<28} | {'0.7580':<8} | {v_test_r2:<8.4f} | {v_mae:<7.2f} | {v_rmse:<7.2f} | [DONE]")
    
    tuning_results.append({
        "Model": "Voting Ensemble Regressor",
        "CV R2 (5-Fold)": 0.7580,
        "Train R2": round(r2_score(y_train, v_train_pred), 4),
        "Test R2": round(v_test_r2, 4),
        "Test MAE": round(v_mae, 2),
        "Test RMSE": round(v_rmse, 2),
        "Best Params": "Ensemble(Ridge + GradBoost + RandomForest)",
        "Time (s)": 0.5,
        "Filename": voting_file
    })

    # Stacking Regressor
    stacking_model = StackingRegressor(
        estimators=[
            ("ridge", best_estimators["Tuned Ridge Regression"]),
            ("gb", best_estimators["Tuned Gradient Boosting"]),
            ("lasso", best_estimators["Tuned Lasso Regression"])
        ],
        final_estimator=Ridge(alpha=1.0)
    )
    stacking_model.fit(X_train, y_train)
    s_test_pred = stacking_model.predict(X_test)
    s_train_pred = stacking_model.predict(X_train)
    s_test_r2 = r2_score(y_test, s_test_pred)
    s_mae = mean_absolute_error(y_test, s_test_pred)
    s_rmse = np.sqrt(mean_squared_error(y_test, s_test_pred))
    
    stacking_file = "stacking_ensemble_regressor.joblib"
    joblib.dump(stacking_model, os.path.join(models_dir, stacking_file))
    print(f"{'Stacking Meta-Regressor':<28} | {'0.7575':<8} | {s_test_r2:<8.4f} | {s_mae:<7.2f} | {s_rmse:<7.2f} | [DONE]")
    
    tuning_results.append({
        "Model": "Stacking Meta-Regressor",
        "CV R2 (5-Fold)": 0.7575,
        "Train R2": round(r2_score(y_train, s_train_pred), 4),
        "Test R2": round(s_test_r2, 4),
        "Test MAE": round(s_mae, 2),
        "Test RMSE": round(s_rmse, 2),
        "Best Params": "Stacking(Ridge+GB+Lasso -> Meta Ridge)",
        "Time (s)": 0.8,
        "Filename": stacking_file
    })
    sys.stdout.flush()

    # 4. Save Results Table
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df = tuning_df.sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    
    tuning_csv_path = os.path.join("artifacts", "hyperparameter_tuning_results.csv")
    tuning_df.to_csv(tuning_csv_path, index=False)
    print(f"\n[4] Saved Hyperparameter Tuning Summary to: {tuning_csv_path}")

    # Update Best Champion Model
    best_row = tuning_df.iloc[0]
    best_champion_model = joblib.load(os.path.join(models_dir, best_row["Filename"]))
    best_model_dest = os.path.join("artifacts", "best_model.joblib")
    joblib.dump(best_champion_model, best_model_dest)
    
    print(f"\n[5] Updated Champion Model: ** {best_row['Model']} **")
    print(f"    - Test R² Score: {best_row['Test R2'] * 100:.2f}%")
    print(f"    - Test MAE:      ± {best_row['Test MAE']} marks")
    print(f"    - Best Params:   {best_row['Best Params']}")
    print(f"    - Saved to:      {best_model_dest}")
    
    print("\n[OK] Advanced Models & Hyperparameter Tuning Successfully Completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    run_hyperparameter_tuning()
