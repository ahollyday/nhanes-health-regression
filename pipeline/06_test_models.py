import pandas as pd
import numpy as np
import joblib
import os
import yaml
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# === Load test data ===
npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_test = npz["X_test"]
y_test = pd.DataFrame(npz["y_test"])

# === Load feature config ===
def load_feature_config():
    with open("../data/config/features.yaml", "r") as f:
        return yaml.safe_load(f)["features"]

features = load_feature_config()
target_names = [f["name"] for f in features if f["role"] == "target"]
log_targets = [f["name"] for f in features if f.get("log_transform") and f["role"] == "target"]
y_test.columns = target_names

# === Inverse log transform ===
def inverse_log_transform(arr, columns):
    df = pd.DataFrame(arr, columns=target_names)
    for col in columns:
        if col in df.columns:
            df[col] = np.expm1(df[col])
    return df

# === Evaluate saved models ===
model_dir = "../models"
results = []
residuals = []
# Read non-empty model filenames
with open("../models/best_models.txt", "r") as f:
    model_files = [line.strip() for line in f if line.strip()]

print(f" Found {len(model_files)} models in best_models.txt:")
for f in model_files:
    print(f"  - {f}")

for filename in model_files:
    name = filename.replace("_", " ").replace(".pkl", "").title()
    print(f"\n Evaluating model: {name}")
    model = joblib.load(os.path.join(model_dir, filename))

    y_pred = model.predict(X_test)
    y_pred_df = inverse_log_transform(y_pred, log_targets)
    y_true_df = inverse_log_transform(y_test.values, log_targets)

    r2 = r2_score(y_true_df, y_pred_df, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true_df, y_pred_df, multioutput='raw_values'))

    results.append({
        "Model": name,
        **{f"R2 - {target_names[i]}": r2[i] for i in range(len(target_names))},
        **{f"RMSE - {target_names[i]}": rmse[i] for i in range(len(target_names))}
    })

    for i, target in enumerate(target_names):
        residuals.append(pd.DataFrame({
            "Model": name,
            "Target": target,
            "Residual": y_true_df.iloc[:, i] - y_pred_df.iloc[:, i],
            "True": y_true_df.iloc[:,i],
            "Predicted": y_pred_df.iloc[:,i] 
        }))

# === Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("../summaries/test_metrics.csv", index=False)
print(" Saved test metrics to ../summaries/test_metrics.csv")

residuals_df = pd.concat(residuals, ignore_index=True)
residuals_df.to_csv("../summaries/test_residuals.csv", index=False)
print(" Saved test residuals to ../summaries/test_residuals.csv")

