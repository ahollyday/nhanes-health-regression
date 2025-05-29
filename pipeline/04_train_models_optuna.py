
import pandas as pd
import joblib
import numpy as np
import os
import yaml
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# === Global config ===
CV_FOLDS = 3
N_TRIALS = 4
MAX_TIME = 18000 # 30 mins
optuna_study_dir = "../optuna_studies"
os.makedirs(optuna_study_dir, exist_ok=True)

# === Load preprocessed data ===
npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = npz["X_train"]
y_train = pd.DataFrame(npz["y_train"])

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

results = []
residuals = []
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

optuna_study_dir = "../optuna_studies"
os.makedirs(optuna_study_dir, exist_ok=True)

def run_study(model_name, objective_func):
    print(f"🔍 Running Optuna for: {model_name}")
    
    db_filename = f"optuna_{model_name.replace(' ', '_').lower()}.db"
    db_path = os.path.join(optuna_study_dir, db_filename)
    
    study = optuna.create_study(
        study_name=model_name.replace(" ", "_").lower(),
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective_func, n_trials=N_TRIALS, timeout=MAX_TIME)
    print(f"✅ Best parameters for {model_name}: {study.best_params}")
    return study.best_params

def train_and_save(model_name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_df = inverse_log_transform(y_pred, log_targets)
    y_true_df = inverse_log_transform(y_train.values, log_targets)

    r2 = r2_score(y_true_df, y_pred_df, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true_df, y_pred_df, multioutput='raw_values'))

    results.append({
        "Model": model_name,
        **{f"R2 - {target_names[i]}": r2[i] for i in range(len(target_names))},
        **{f"RMSE - {target_names[i]}": rmse[i] for i in range(len(target_names))}
    })

    for i, target in enumerate(target_names):
        residuals.append(pd.DataFrame({
            "Model": model_name,
            "Target": target,
            "Residual": y_true_df.iloc[:, i] - y_pred_df.iloc[:, i]
        }))

    model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model to {model_path}")
    return model_path

with open("../models/best_models.txt", "w") as f:
    print("\n🚀 Training model: Linear Regression")
    linear_model = MultiOutputRegressor(LinearRegression())
    path = train_and_save("Linear Regression", linear_model)
    f.write(f"{os.path.basename(path)}\n")

    def objective_rf(trial):
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            n_jobs=1, random_state=42))
        return cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2').mean()

    best_params = run_study("Random Forest", objective_rf)
    rf_model = MultiOutputRegressor(RandomForestRegressor(**best_params, random_state=42, n_jobs=1))
    path = train_and_save("Random Forest", rf_model)
    f.write(f"{os.path.basename(path)}\n")

    def objective_gb(trial):
        model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.3),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=42))
        return cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2').mean()

    best_params = run_study("Gradient Boosting", objective_gb)
    gb_model = MultiOutputRegressor(GradientBoostingRegressor(**best_params, random_state=42))
    path = train_and_save("Gradient Boosting", gb_model)
    f.write(f"{os.path.basename(path)}\n")

    def objective_knn(trial):
        model = MultiOutputRegressor(KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 5, 50),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"])))
        return cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2').mean()

    best_params = run_study("KNN", objective_knn)
    knn_model = MultiOutputRegressor(KNeighborsRegressor(**best_params))
    path = train_and_save("KNN", knn_model)
    f.write(f"{os.path.basename(path)}\n")

    def objective_svr(trial):
        model = MultiOutputRegressor(SVR(
            C=trial.suggest_float("C", 0.1, 100),
            epsilon=trial.suggest_float("epsilon", 0.01, 1.0),
            gamma=trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
            kernel="rbf"))
        return cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2').mean()

    best_params = run_study("SVR", objective_svr)
    svr_model = MultiOutputRegressor(SVR(**best_params))
    path = train_and_save("SVR", svr_model)
    f.write(f"{os.path.basename(path)}\n")

    def objective_xgb(trial):
        model = MultiOutputRegressor(XGBRegressor(
            objective='reg:squarederror',
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.3),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0, 5.0),
            random_state=42,
            n_jobs=1))
        return cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2').mean()

    best_params = run_study("XGBoost", objective_xgb)
    xgb_model = MultiOutputRegressor(XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=1))
    path = train_and_save("XGBoost", xgb_model)
    f.write(f"{os.path.basename(path)}\n")

results_df = pd.DataFrame(results)
results_df.to_csv("../summaries/train_metrics.csv", index=False)
print("📊 Saved training metrics to ../summaries/train_metrics.csv")

residuals_df = pd.concat(residuals, ignore_index=True)
residuals_df.to_csv("../summaries/train_residuals.csv", index=False)
print("📉 Saved training residuals to ../summaries/train_residuals.csv")
