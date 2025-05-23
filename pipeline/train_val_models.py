import pandas as pd
import joblib
import numpy as np
import os
import yaml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

# === Load preprocessed data ===
npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = npz["X_train"]
X_test = npz["X_test"]
y_train = pd.DataFrame(npz["y_train"])
y_test = pd.DataFrame(npz["y_test"])
feature_names = npz["feature_names"]

# === Load target names dynamically ===
def load_target_names():
    with open("../data/config/features.yaml", "r") as f:
        features = yaml.safe_load(f)["features"]
    return [f["name"] for f in features if f["role"] == "target"]

target_names = load_target_names()
y_train.columns = target_names
y_test.columns = target_names

# === Define models and parameter distributions ===
model_specs = {
    "Linear Regression": {
        "model": MultiOutputRegressor(LinearRegression()),
        "params": None
    },
    "Random Forest": {
        "model": MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        "params": {
            "estimator__n_estimators": randint(100, 500),
            "estimator__max_depth": randint(3, 20),
            "estimator__min_samples_split": randint(2, 10)
        }
    },
    "Gradient Boosting": {
        "model": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        "params": {
            "estimator__n_estimators": randint(100, 300),
            "estimator__learning_rate": uniform(0.01, 0.3),
            "estimator__max_depth": randint(3, 10)
        }
    },
    "KNN": {
        "model": MultiOutputRegressor(KNeighborsRegressor()),
        "params": {
            "estimator__n_neighbors": randint(3, 20),
            "estimator__weights": ["uniform", "distance"]
        }
    },
    "SVR": {
        "model": MultiOutputRegressor(SVR()),
        "params": {
            "estimator__C": uniform(0.1, 100),
            "estimator__epsilon": uniform(0.01, 1.0),
            "estimator__kernel": ["rbf"]
        }
    }
}

results = []
residuals = []
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

for name, spec in model_specs.items():
    print(f"\n🚀 Training model: {name}")
    model = spec["model"]
    params = spec.get("params")

    if params:
        clf = RandomizedSearchCV(model, params, n_iter=25, cv=3, scoring="r2", verbose=1, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    for split, X, y_true in [("train", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = best_model.predict(X)
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

        results.append({
            "Model": name,
            "Split": split,
            **{f"R2 - {target_names[i]}": r2[i] for i in range(len(target_names))},
            **{f"RMSE - {target_names[i]}": rmse[i] for i in range(len(target_names))}
        })

        for i, target in enumerate(target_names):
            residuals.append(pd.DataFrame({
                "Model": name,
                "Split": split,
                "Target": target,
                "Residual": y_true.iloc[:, i] - y_pred[:, i]
            }))

    joblib.dump(best_model, os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl"))
    print(f"✅ Saved model to {model_dir}/{name.replace(' ', '_').lower()}.pkl")

results_df = pd.DataFrame(results)
results_df.to_csv("../summaries/model_metrics.csv", index=False)
print("\n📊 Saved model performance summary to ../summaries/model_metrics.csv")

residuals_df = pd.concat(residuals, ignore_index=True)
residuals_df.to_csv("../summaries/residuals.csv", index=False)
print("📉 Saved residuals to ../summaries/residuals.csv")

