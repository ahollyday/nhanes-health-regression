import pandas as pd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from joblib import load

# === Load data ===
npz = np.load("../data/processed/train_test_data.npz", allow_pickle=True)
X = pd.DataFrame(np.concatenate([npz["X_train"], npz["X_test"]], axis=0))
y = pd.DataFrame(np.vstack([npz["y_train"], npz["y_test"]]))

# === Load target names ===
def load_target_names():
    with open("../data/config/features.yaml", "r") as f:
        features = yaml.safe_load(f)["features"]
    return [f["name"] for f in features if f["role"] == "target"]

target_names = load_target_names()
y.columns = target_names

# === Load best model per name ===
model_dir = "../models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
model_specs = {}
for f in model_files:
    name = f.replace("_", " ").replace(".pkl", "").title()
    if name == "Linear Regression":
        name = "Baseline Linear Model"
    model_specs[name] = load(os.path.join(model_dir, f))

# === Evaluation ===
cv_values = [2, 3, 5, 7, 10, 20]
cv_results = []

for name, model in model_specs.items():
    print(f"\n🔍 Evaluating CV effect for: {name}")
    for cv in cv_values:
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        rmse_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))),
            n_jobs=-1
        )

        cv_results.append({
            "Model": name,
            "CV Folds": cv,
            "Mean R2": np.mean(r2_scores),
            "Std R2": np.std(r2_scores),
            "Mean RMSE": np.mean(rmse_scores),
            "Std RMSE": np.std(rmse_scores)
        })

# === Save results ===
cv_results_df = pd.DataFrame(cv_results)
os.makedirs("../summaries", exist_ok=True)
cv_results_df.to_csv("../summaries/cv_sweep_results.csv", index=False)
print("\n📈 Saved CV sweep results to ../summaries/cv_sweep_results.csv")

# === Plotting ===
sns.set(style="white")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
lineplot_r2 = sns.lineplot(data=cv_results_df, x="CV Folds", y="Mean R2", hue="Model", marker="o", ax=axes[0])

axes[0].set_ylabel("Mean R²")
axes[0].set_xlabel("Number of CV Folds")
axes[0].grid(False)

sns.lineplot(data=cv_results_df, x="CV Folds", y="Mean RMSE", hue="Model", marker="o", ax=axes[1], legend=False)

axes[1].set_ylabel("Mean RMSE")
axes[1].set_xlabel("Number of CV Folds")
axes[1].grid(False)

# Shared legend
handles, labels = lineplot_r2.get_legend_handles_labels()
fig.legend(handles, labels, title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("../figures/cv_sweep_summary.png", dpi=300)
print("📊 Saved CV sweep plot to ../figures/cv_sweep_summary.png")
# plt.show()  # Disabled to avoid popup

