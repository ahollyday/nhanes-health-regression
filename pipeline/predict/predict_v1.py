
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Paths
PREPROCESSOR_PATH = "../../models/preprocessor.joblib"
PREDICT_DATA_PATH = "../../data/predict_processed/clean_data_2017_2018.csv"
MODEL_DIR = "../../models"
OUTPUT_DIR = "../../figures/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data and preprocessor
df = pd.read_csv(PREDICT_DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Load feature config
import yaml
with open("../../data/config/features.yaml", 'r') as f:
    features = yaml.safe_load(f)["features"]

numeric_features = [f["name"] for f in features if f["type"] in ["numeric", "numerical"] and f["role"] == "feature"]
categorical_features = [f["name"] for f in features if f["type"] == "categorical" and f["role"] == "feature"]
target_features = [f["name"] for f in features if f["role"] == "target"]
log_transform_targets = [f["name"] for f in features if f["role"] == "target" and f.get("log_transform", False)]

X = df[numeric_features + categorical_features].copy()
y_true = df[target_features].copy()

# Apply log transform to targets if needed (for true comparison)
y_true[log_transform_targets] = np.log1p(y_true[log_transform_targets])

# Transform features
X_transformed = preprocessor.transform(X)

# Evaluate each model
model_names = ["random_forest", "xgboost", "gradient_boosting"]
for model_name in model_names:
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_transformed)

        for i, target in enumerate(target_features):
            plt.figure(figsize=(6, 6), dpi=300)
            plt.scatter(y_true[target], y_pred[:, i], alpha=0.6, edgecolor='k')
            plt.xlabel(f"True {target}")
            plt.ylabel(f"Predicted {target}")
            plt.title(f"{model_name.replace('_', ' ').title()} - {target}")
            plt.plot([y_true[target].min(), y_true[target].max()],
                     [y_true[target].min(), y_true[target].max()],
                     'r--', lw=2)
            plt.grid(True)
            plt.tight_layout()
            fig_path = os.path.join(OUTPUT_DIR, f"{model_name}_{target}_true_vs_pred.png")
            plt.savefig(fig_path)
            plt.close()

            # Save residual plot
            residuals = y_true[target] - y_pred[:, i]
            plt.figure(figsize=(6, 4), dpi=300)
            sns.histplot(residuals, bins=30, kde=True, color="steelblue")
            plt.title(f"{model_name.replace('_', ' ').title()} Residuals - {target}")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{target}_residuals.png"))
            plt.close()

            # Print metrics
            mse = mean_squared_error(y_true[target], y_pred[:, i])
            r2 = r2_score(y_true[target], y_pred[:, i])
            print(f"✅ {model_name} - {target}: R² = {r2:.3f}, MSE = {mse:.3f}")
    except Exception as e:
        print(f"⚠️ Skipped {model_name} due to error: {e}")
