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
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import issparse
import sys

# === Load preprocessed data ===
npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = npz["X_train"]
y_train = pd.DataFrame(npz["y_train"])

#if issparse(X_train):
#    print("❌ X_train is a sparse matrix. Exiting to prevent LinearRegression failure.")
#    exit(1)
#    sys.exit("exiting")

#sys.exit("exiting")

# === Load target names dynamically ===
def load_feature_config():
    with open("../data/config/features.yaml", "r") as f:
        return yaml.safe_load(f)["features"]

def load_target_names():
    features = load_feature_config()
    return [f["name"] for f in features if f["role"] == "target"]

def get_log_transform_targets():
    features = load_feature_config()
    return [f["name"] for f in features if f.get("log_transform") and f["role"] == "target"]

def inverse_log_transform(arr, columns):
    df = pd.DataFrame(arr, columns=target_names)
    for col in columns:
        if col in df.columns:
            df[col] = np.expm1(df[col])
    return df

target_names = load_target_names()
log_targets = get_log_transform_targets()
y_train.columns = target_names

# === Define models and expanded parameter distributions ===
model_specs = {
    "Linear Regression": {
        "model": MultiOutputRegressor(LinearRegression()),
        "params": None
    },
    "Random Forest": {
        "model": MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        "params": {
            "estimator__n_estimators": randint(100, 1000),
            "estimator__max_depth": randint(3, 30),
            "estimator__min_samples_split": randint(2, 20),
            "estimator__min_samples_leaf": randint(1, 10),
            "estimator__max_features": ["sqrt", "log2"]
        }
    },
    "Gradient Boosting": {
        "model": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        "params": {
            "estimator__n_estimators": randint(100, 1000),
            "estimator__learning_rate": uniform(0.001, 0.3),
            "estimator__max_depth": randint(3, 10),
            "estimator__subsample": uniform(0.5, 0.5),
            "estimator__min_samples_leaf": randint(1, 10)
        }
    },
    "KNN": {
        "model": MultiOutputRegressor(KNeighborsRegressor()),
        "params": {
            "estimator__n_neighbors": randint(5, 50),
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
    },
    "XGBoost": {
        "model": MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42)),
        "params": {
            "estimator__n_estimators": randint(100, 1000),
            "estimator__max_depth": randint(3, 15),
            "estimator__learning_rate": uniform(0.001, 0.3),
            "estimator__subsample": uniform(0.5, 0.5),
            "estimator__colsample_bytree": uniform(0.5, 0.5)
        }
    }
}

results = []
residuals = []
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

with open("../models/best_models.txt", "w") as f:
    for name, spec in model_specs.items():
        print(f"\n🚀 Training model: {name}")
        model = spec["model"]
        params = spec.get("params")

        if params:
            clf = RandomizedSearchCV(model, params, n_iter=5, cv=10, scoring="r2", verbose=1, n_jobs=3, random_state=42)
            clf.fit(X_train, y_train)
            best_model = clf.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        y_pred = best_model.predict(X_train)
        y_pred_df = inverse_log_transform(y_pred, log_targets)
        y_true_df = inverse_log_transform(y_train.values, log_targets)

        r2 = r2_score(y_true_df, y_pred_df, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_true_df, y_pred_df, multioutput='raw_values'))

        results.append({
            "Model": name,
            **{f"R2 - {target_names[i]}": r2[i] for i in range(len(target_names))},
            **{f"RMSE - {target_names[i]}": rmse[i] for i in range(len(target_names))}
        })

        print(f"📈 Mean R² across targets: {np.mean(r2):.3f}")

        for i, target in enumerate(target_names):
            residuals.append(pd.DataFrame({
                "Model": name,
                "Target": target,
                "Residual": y_true_df.iloc[:, i] - y_pred_df.iloc[:, i]
            }))

        model_path = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(best_model, model_path)
        f.write(f"{os.path.basename(model_path)}\n")
        print(f"✅ Saved model to {model_path}")

# === Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("../summaries/train_metrics.csv", index=False)
print("📊 Saved training metrics to ../summaries/train_metrics.csv")

residuals_df = pd.concat(residuals, ignore_index=True)
residuals_df.to_csv("../summaries/train_residuals.csv", index=False)
print("📉 Saved training residuals to ../summaries/train_residuals.csv")
