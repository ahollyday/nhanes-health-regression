import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import yaml

# === Paths ===
PREDICT_DATA_PATH = "../../data/predict_processed/clean_data_2017_2018.csv"
PREPROCESSOR_PATH = "../../models/preprocessor.joblib"
MODEL_DIR = "../../models"
OUTPUT_FIG_PATH = "../../figures/predictions/true_vs_predicted_all_models.png"

# === Models to Evaluate ===
model_files = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "KNN": "knn.pkl",
    "Linear Regression": "linear_regression.pkl",
    "SVR": "svr.pkl",
    "XGBoost": "xgboost.pkl"
}

# === Load Data and Preprocessor ===
df = pd.read_csv(PREDICT_DATA_PATH)
with open("../../data/config/features.yaml", "r") as f:
    features = yaml.safe_load(f)["features"]
X_cols = [f["name"] for f in features if f["role"] == "feature"]
y_cols = [f["name"] for f in features if f["role"] == "target"]
log_targets = [f["name"] for f in features if f["role"] == "target" and f.get("log_transform", False)]

X = df[X_cols].copy()
y_true = df[y_cols].copy()

preprocessor = load(PREPROCESSOR_PATH)
X_transformed = preprocessor.transform(X)

# === Predict with Each Model ===
predictions = {}

for model_name, filename in model_files.items():
    model_path = os.path.join(MODEL_DIR, filename)
    try:
        model = load(model_path)
        y_pred = model.predict(X_transformed)
        if y_pred.shape[1] != len(y_cols):
            continue
        df_pred = pd.DataFrame(y_pred, columns=y_cols)
        for col in log_targets:
            if col in df_pred:
                df_pred[col] = np.expm1(df_pred[col])
        predictions[model_name] = df_pred
    except Exception as e:
        print(f"⚠️ Skipping {model_name} due to error: {e}")

# === Plot Grid of Scatterplots ===
n_targets = len(y_cols)
n_models = len(predictions)

fig, axes = plt.subplots(n_targets, n_models, figsize=(4 * n_models, 4 * n_targets), dpi=300)
if n_targets == 1:
    axes = np.expand_dims(axes, axis=0)
if n_models == 1:
    axes = np.expand_dims(axes, axis=1)

for row_idx, target in enumerate(y_cols):
    for col_idx, (model_name, df_pred) in enumerate(predictions.items()):
        ax = axes[row_idx, col_idx]
        ax.scatter(y_true[target], df_pred[target], s=10, alpha=0.6)
        ax.plot([y_true[target].min(), y_true[target].max()],
                [y_true[target].min(), y_true[target].max()],
                'r--', lw=1)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"{model_name}\nTarget: {target}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_FIG_PATH), exist_ok=True)
plt.savefig(OUTPUT_FIG_PATH)
plt.close()
print(f"✅ Saved figure to {OUTPUT_FIG_PATH}")

