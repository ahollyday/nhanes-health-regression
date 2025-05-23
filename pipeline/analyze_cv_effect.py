import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# === Load models
model_dir = "../models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
model_specs = {}
for f in model_files:
    name = f.replace("_", " ").replace(".pkl", "").title()
    if name == "Linear Regression":
        name = "Baseline Linear Model"
    model_specs[name] = load(os.path.join(model_dir, f))

# === Load data
data = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]

# === Cross-validation analysis
folds = [2, 3, 5, 7, 10, 15, 20]
results = []

for name, model in model_specs.items():
    print(f"\n🔍 Evaluating CV effect for: {name}")
    for cv in folds:
        r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1))
        results.append({
            "Model": name,
            "CV": cv,
            "R2 Mean": np.mean(r2_scores),
            "R2 Std": np.std(r2_scores),
            "RMSE Mean": np.mean(rmse_scores),
            "RMSE Std": np.std(rmse_scores)
        })

# === Save results
df = pd.DataFrame(results)

# === Plot
sns.set(style="white")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
r2_ax, rmse_ax = axes

# R² plot
sns.lineplot(data=df, x="CV", y="R2 Mean", hue="Model", marker="o", ax=r2_ax)
r2_ax.set_ylabel("Mean R²")
r2_ax.set_xlabel("Number of CV Folds")

# RMSE plot
sns.lineplot(data=df, x="CV", y="RMSE Mean", hue="Model", marker="o", ax=rmse_ax)
rmse_ax.set_ylabel("Mean RMSE")
rmse_ax.set_xlabel("Number of CV Folds")

# Move legend to one plot only
r2_ax.legend(title="Model", fontsize=8)
rmse_ax.get_legend().remove()

plt.tight_layout()
os.makedirs("../figures/evaluation", exist_ok=True)
plt.savefig("../figures/evaluation/cv_fold_analysis.png", dpi=300)
print("📊 Saved CV analysis plot to ../figures/evaluation/cv_fold_analysis.png")

