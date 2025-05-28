import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load

# Set Seaborn style
sns.set(style='whitegrid')

# === CONFIG ===
PREDICTIONS_PATH = "../summaries/predictions_2017_2018.csv"
MODEL_DIR = "../models"
OUTPUT_DIR = "../figures/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load predictions ===
df = pd.read_csv(PREDICTIONS_PATH)
true_cols = [col for col in df.columns if col.startswith("true_")]
pred_cols = [col for col in df.columns if col.startswith("pred_")]

y_true = df[true_cols].rename(columns=lambda x: x.replace("true_", ""))
y_pred = df[pred_cols].rename(columns=lambda x: x.replace("pred_", ""))

# === 1. Actual vs Predicted ===
for col in y_true.columns:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[col], y_pred[col], alpha=0.4)
    lims = [min(y_true[col].min(), y_pred[col].min()), max(y_true[col].max(), y_pred[col].max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted: {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted_{col}.png")
    plt.close()

# === 2. Residual Histograms ===
residuals = y_true - y_pred
for col in residuals.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals[col], kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Residuals: {col}")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/residuals_{col}.png")
    plt.close()

# === 3. Model Comparison (optional summary from predictions file) ===
# Expect predictions file to contain model-wise R² and RMSE columns if available
model_metrics_path = "../summaries/model_metrics.csv"
if os.path.exists(model_metrics_path):
    metrics_df = pd.read_csv(model_metrics_path)
    melted = metrics_df.melt(id_vars="Model", value_name="Score", var_name="Metric")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="Score", y="Model", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_barplot.png")
    plt.close()

# === 4. Feature Importances (tree models only) ===
tree_models = ["random_forest.pkl", "xgboost.pkl", "gradient_boosting.pkl"]
for model_file in tree_models:
    model_path = Path(MODEL_DIR) / model_file
    if not model_path.exists():
        continue

    try:
        model = load(model_path)
        reg = model.named_steps['regressor'].estimators_[0]
        importances = reg.feature_importances_
        features = model.named_steps['preprocessing'].get_feature_names_out()

        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(importances)), importances[sorted_idx])
        plt.yticks(range(len(importances)), np.array(features)[sorted_idx])
        plt.xlabel("Importance")
        plt.title(f"Feature Importances: {model_file.replace('.pkl', '')}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importances_{model_file.replace('.pkl', '')}.png")
        plt.close()

    except Exception as e:
        print(f"⚠️ Skipped {model_file} due to error: {e}")

