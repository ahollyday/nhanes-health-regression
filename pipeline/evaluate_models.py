import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from matplotlib.lines import Line2D

# === Load data and config ===
train_metrics = pd.read_csv("../summaries/train_metrics.csv")
test_metrics = pd.read_csv("../summaries/test_metrics.csv")
train_residuals = pd.read_csv("../summaries/train_residuals.csv")
test_residuals = pd.read_csv("../summaries/test_residuals.csv")

with open("../data/config/features.yaml", "r") as f:
    features = yaml.safe_load(f)["features"]

targets = [f["name"] for f in features if f["role"] == "target"]
target_units = {f["name"]: f.get("unit", "") for f in features if f["role"] == "target"}
log_targets = [f["name"] for f in features if f.get("log_transform")]

# === Plot scatter of Train vs Test metrics (R2 and RMSE) ===
metrics = ["R2", "RMSE"]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

colors = sns.color_palette(n_colors=len(train_metrics))

legend_elements = []
for i, metric in enumerate(metrics):
    ax = axes[i]
    train_cols = [col for col in train_metrics.columns if col.startswith(metric)]
    test_cols = [col for col in test_metrics.columns if col.startswith(metric)]

    for j, model in enumerate(train_metrics["Model"]):
        train_values = train_metrics.loc[j, train_cols].values
        test_values = test_metrics.loc[j, test_cols].values
        mean_train = np.mean(train_values)
        mean_test = np.mean(test_values)
        ax.scatter(mean_train, mean_test, color=colors[j])
        if i == 0:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=model,
                                          markerfacecolor=colors[j], markersize=6))

    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c="gray")
    ax.set_xlabel(f"Train {metric}")
    ax.set_ylabel(f"Test {metric}")
    ax.grid(False)

fig.legend(handles=legend_elements, loc='center right', frameon=False)
plt.tight_layout(rect=[0, 0, 0.95, 1])
os.makedirs("../figures/evaluation", exist_ok=True)
plt.savefig("../figures/evaluation/train_test_metrics_scatter.png", dpi=300)
print("✅ Saved train/test scatter comparison to ../figures/evaluation/train_test_metrics_scatter.png")

# === Residual overlay plots ===
def plot_residual_overlay(residual_df, split, out_path, title):
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4))
    if len(targets) == 1:
        axes = [axes]

    for i, target in enumerate(targets):
        ax = axes[i]
        for model_name in residual_df["Model"].unique():
            subset = residual_df[(residual_df["Model"] == model_name) & (residual_df["Target"] == target)]
            sns.histplot(subset["Residual"], kde=False, bins=50, ax=ax, label=model_name,
                         element="step", fill=False)

        ax.axvline(0, linestyle="--", color="black")
        unit = f" ({target_units[target]})" if target_units[target] else ""
        ax.set_xlabel(f"{target}{unit}")
        ax.set_ylabel("Density")
        ax.grid(False)

    fig.legend(handles=[Line2D([0], [0], color=sns.color_palette()[i], label=model)
                        for i, model in enumerate(residual_df["Model"].unique())],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved {split} residuals overlay to {out_path}")

plot_residual_overlay(train_residuals, "Train", "../figures/evaluation/train_residuals_overlay.png", "Train Residuals")
plot_residual_overlay(test_residuals, "Test", "../figures/evaluation/test_residuals_overlay.png", "Test Residuals")

# === CV analysis directly in script ===
from joblib import load
model_dir = "../models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
model_specs = {}
for f in model_files:
    name = f.replace("_", " ").replace(".pkl", "").title()
    if name == "Linear Regression":
        name = "Baseline Linear Model"
    model_specs[name] = load(os.path.join(model_dir, f))

data = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]

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

df = pd.DataFrame(results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
r2_ax, rmse_ax = axes

sns.lineplot(data=df, x="CV", y="R2 Mean", hue="Model", marker="o", ax=r2_ax)
r2_ax.set_ylabel("Mean R²")
r2_ax.set_xlabel("Number of CV Folds")
r2_ax.grid(False)

sns.lineplot(data=df, x="CV", y="RMSE Mean", hue="Model", marker="o", ax=rmse_ax)
rmse_ax.set_ylabel("Mean RMSE")
rmse_ax.set_xlabel("Number of CV Folds")
rmse_ax.grid(False)

r2_ax.legend(title="Model", fontsize=8)
rmse_ax.get_legend().remove()

plt.tight_layout()
plt.savefig("../figures/evaluation/cv_fold_analysis.png", dpi=300)
print("📊 Saved CV analysis plot to ../figures/evaluation/cv_fold_analysis.png")

