import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import yaml
import matplotlib.patches as mpatches
import gc
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate

# === Load config ===
def load_feature_config():
    with open("../data/config/features.yaml", "r") as f:
        return yaml.safe_load(f)["features"]

features = load_feature_config()
target_names = [f["name"] for f in features if f["role"] == "target"]
target_units = {f["name"]: f.get("unit", "") for f in features if f["role"] == "target"}
target_display_names = {f["name"]: f.get("display_name", f["name"]) for f in features if f["role"] == "target"}

# === Standardize model names ===
def standardize_model_name(name):
    name = name.strip().lower()
    if name == "linear regression":
        return "Baseline Linear Model"
    return name.title()

# === Load metrics and residuals ===
train_metrics = pd.read_csv("../summaries/train_metrics.csv")
test_metrics = pd.read_csv("../summaries/test_metrics.csv")
train_res = pd.read_csv("../summaries/train_residuals.csv")
test_res = pd.read_csv("../summaries/test_residuals.csv")

for df in [train_metrics, test_metrics, train_res, test_res]:
    df["Model"] = df["Model"].apply(standardize_model_name)

# === Parse metrics ===
def extract_metric(df, metric):
    return df.filter(like=metric).set_index(df["Model"])

r2_train = extract_metric(train_metrics, "R2")
r2_test = extract_metric(test_metrics, "R2")
rmse_train = extract_metric(train_metrics, "RMSE")
rmse_test = extract_metric(test_metrics, "RMSE")

# === Define consistent tab10 color palette ===
all_models = sorted(set(train_metrics["Model"]) | set(test_metrics["Model"]) |
                    set(train_res["Model"]) | set(test_res["Model"]))
palette = sns.color_palette("tab10", n_colors=len(all_models))
model_colors = dict(zip(all_models, palette))

# === Plot R^2 and RMSE comparison ===
sns.set(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

for (train_df, test_df, metric, ax) in [
    (r2_train, r2_test, "R^2", axes[0]),
    (rmse_train, rmse_test, "RMSE", axes[1])
]:
    models = train_df.index.tolist()
    #merged = pd.DataFrame({
    #    "Train": train_df.mean(axis=1).values,
    #    "Test": test_df.mean(axis=1).values,
    #    "Model": models
    #})

    # Normalize RMSE columns by their column-wise max
    if metric == "RMSE":
        train_df_norm = train_df / train_df.max(axis=0)
        test_df_norm = test_df / test_df.max(axis=0)
    else:
        train_df_norm = train_df
        test_df_norm = test_df    

    merged = pd.DataFrame({
        "Train": train_df_norm.mean(axis=1).values,
        "Test": test_df_norm.mean(axis=1).values,
        "Model": models
    })

    sns.scatterplot(
        data=merged, x="Train", y="Test", hue="Model", ax=ax,
        palette=model_colors, s=50, edgecolor='white')
    min_val, max_val = merged[["Train", "Test"]].min().min(), merged[["Train", "Test"]].max().max()
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    ax.set_xlabel(f"Train {metric}")
    ax.set_ylabel(f"Test {metric}")
    ax.grid(False)

axes[1].legend(title="Model", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
axes[0].get_legend().remove()
plt.tight_layout()
os.makedirs("../figures/evaluation", exist_ok=True)
plt.savefig("../figures/evaluation/train_test_metrics_comparison.png", dpi=300)

# === Plot R^2 and RMSE comparison with per-target values ===
sns.set(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

for (train_df, test_df, metric, ax) in [
    (r2_train, r2_test, "R^2", axes[0]),
    (rmse_train, rmse_test, "RMSE", axes[1])
]:
    models = train_df.index.tolist()

    if metric == "RMSE":
        train_df_norm = train_df / train_df.max(axis=0)
        test_df_norm = test_df / test_df.max(axis=0)
    else:
        train_df_norm = train_df
        test_df_norm = test_df

    # === Plot mean metrics ===
    merged_mean = pd.DataFrame({
        "Train": train_df_norm.mean(axis=1).values,
        "Test": test_df_norm.mean(axis=1).values,
        "Model": models,
        "Target": "Mean"
    })

    # === Melt per-target values ===
    # Clean up target names for plotting
    clean_target_labels = {
    t: target_display_names.get(t.replace("RMSE - ", "").replace("R2 - ", ""), t.replace("RMSE - ", "").replace("R2 - ", ""))
    for t in train_df_norm.columns
}


    melted_train = train_df_norm.reset_index().melt(id_vars="Model", var_name="Target", value_name="Train")
    melted_test = test_df_norm.reset_index().melt(id_vars="Model", var_name="Target", value_name="Test")

    # Replace technical names with clean labels
    melted_train["Target"] = melted_train["Target"].map(clean_target_labels)
    melted_test["Target"] = melted_test["Target"].map(clean_target_labels)

    #melted_train = train_df_norm.reset_index().melt(id_vars="Model", var_name="Target", value_name="Train")
    #melted_test = test_df_norm.reset_index().melt(id_vars="Model", var_name="Target", value_name="Test")
    merged_targets = pd.merge(melted_train, melted_test, on=["Model", "Target"])

    # === Combine mean + target-level data ===
    merged_all = pd.concat([merged_targets, merged_mean], axis=0)

    # === Plot ===
    sns.scatterplot(
        data=merged_all, x="Train", y="Test", hue="Model", style="Target",
        palette=model_colors, ax=ax, s=50, edgecolor='white'
    )

    min_val, max_val = merged_all[["Train", "Test"]].min().min(), merged_all[["Train", "Test"]].max().max()
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    ax.set_xlabel(f"Train {metric}")
    ax.set_ylabel(f"Test {metric}")
    ax.grid(False)

axes[1].legend(title="", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
axes[0].get_legend().remove()
plt.tight_layout()
os.makedirs("../figures/evaluation", exist_ok=True)
plt.savefig("../figures/evaluation/train_test_metrics_comparison_targets.png", dpi=300)

# === Residual overlay ===
def plot_residual_overlay(df, split):
    g = sns.FacetGrid(df, col="Target", sharex=False, sharey=False, col_wrap=2, height=3.5, aspect=1.4)
    g.map_dataframe(sns.histplot, x="Residual", hue="Model", palette=model_colors,
                    element="step", stat="density", common_norm=False, fill=False, linewidth=1)

    for ax, target in zip(g.axes.flat, g.col_names):
        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
        ax.grid(False)
        ax.set_ylabel("Density")
        display_label = target_display_names.get(target, target)
        unit = target_units.get(target, "")
        ax.set_xlabel(f"{display_label} ({unit})" if unit else display_label)

    g.set_titles("")
    for ax in g.axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()

    handles = [mpatches.Patch(color=model_colors[m], label=m) for m in sorted(df["Model"].unique())]
    g.fig.legend(handles, [h.get_label() for h in handles], title="Model",
                 loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    g.fig.subplots_adjust(top=0.88, bottom=0.28)
    g.fig.suptitle(f"{split} Residuals", fontsize=14)
    g.savefig(f"../figures/evaluation/{split.lower()}_residuals_overlay.png", dpi=300)

plot_residual_overlay(train_res, "Train")
plot_residual_overlay(test_res, "Test")

# === CV analysis ===
from sklearn.model_selection import cross_validate

model_dir = "../models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X_train = npz["X_train"]
y_train = npz["y_train"]

folds = [3, 5, 10, 20]
results = []

for f in model_files:
    name = standardize_model_name(f.replace("_", " ").replace(".pkl", ""))
    model = joblib.load(os.path.join(model_dir, f))
    print(f"\n🔍 Evaluating CV effect for: {name}")

    for cv in folds:
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"r2": "r2", "rmse": "neg_mean_squared_error"},
            n_jobs=1, return_train_score=False
        )
        results.append({
            "Model": name,
            "CV": cv,
            "R2 Mean": np.mean(scores["test_r2"]),
            "R2 Std": np.std(scores["test_r2"]),
            "RMSE Mean": np.mean(np.sqrt(-scores["test_rmse"])),
            "RMSE Std": np.std(np.sqrt(-scores["test_rmse"]))
        })

    del model
    gc.collect()

cv_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
sns.lineplot(data=cv_df, x="CV", y="R2 Mean", hue="Model", marker="o", ax=axes[0], palette=model_colors)
axes[0].set_ylabel("Mean R²")
axes[0].set_xlabel("Number of CV Folds")
axes[0].grid(False)

sns.lineplot(data=cv_df, x="CV", y="RMSE Mean", hue="Model", marker="o", ax=axes[1], palette=model_colors)
axes[1].set_ylabel("Mean RMSE")
axes[1].set_xlabel("Number of CV Folds")
axes[1].grid(False)
axes[1].get_legend().remove()
axes[0].legend(title="Model", fontsize=8)

plt.tight_layout()
plt.savefig("../figures/evaluation/cv_fold_analysis.png", dpi=300)
print(" Saved CV analysis plot to ../figures/evaluation/cv_fold_analysis.png")

