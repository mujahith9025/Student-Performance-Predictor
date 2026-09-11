import os
import sys
import time
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def train_all_models():
    print("=" * 65)
    print("        STUDENT PERFORMANCE PREDICTOR - MODEL TRAINING        ")
    print("=" * 65)
    sys.stdout.flush()

    # 1. Load Processed Training Data
    processed_dir = os.path.join("data", "processed")
    X_train_path = os.path.join(processed_dir, "X_train.npy")
    y_train_path = os.path.join(processed_dir, "y_train.npy")

    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("Processed training data not found! Please run Phase 3 preprocessing first.")

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    print(f"\n[1] Loaded Training Data:")
    print(f"    - X_train shape: {X_train.shape} (800 student samples, 14 features)")
    print(f"    - y_train shape: {y_train.shape}")
    sys.stdout.flush()

    # 2. Define Model Zoo
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, n_jobs=1)
    }

    # 3. Train Each Model
    models_dir = os.path.join("artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n[2] Training Machine Learning Models:\n")
    trained_models = {}

    for name, model in models.items():
        start_time = time.time()
        
        # Fit model on training set
        model.fit(X_train, y_train)
        elapsed_time = (time.time() - start_time) * 1000  # in ms
        
        # Save individual model checkpoint
        safe_filename = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        save_path = os.path.join(models_dir, safe_filename)
        joblib.dump(model, save_path)
        
        trained_models[name] = model
        print(f" [DONE] {name:<25} (Time: {elapsed_time:5.1f}ms | Saved: {safe_filename})")
        sys.stdout.flush()

    print(f"\n[3] All {len(trained_models)} models trained and saved to '{models_dir}'.")
    print("\n[OK] Phase 4 Model Training Successfully Completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    train_all_models()
