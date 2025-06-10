import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re
import yaml
from collections import defaultdict

# === Load preprocessor ===
preprocessor = joblib.load("../models/preprocessor.joblib")
feature_names = preprocessor.get_feature_names_out()

# === Load actual target names from config ===
def get_target_names(config_path="../data/config/features.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return [f["name"] for f in config["features"] if f["role"] == "target"]

target_names = get_target_names()

# === Collapse one-hot encoded feature names ===
def collapse_onehot_feature_names(feature_names):
    grouped = defaultdict(list)
    for name in feature_names:
        if "cat__" in name:
            base = re.sub(r"_\d+(\.0)?$", "", name.split("cat__")[1])
        elif "num__" in name:
            base = name.split("num__")[1]
        else:
            base = name
        grouped[base].append(name)
    return grouped

# === Load and evaluate each model ===
model_dir = "../models"
os.makedirs("../figures/feature_importance", exist_ok=True)
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and f != "preprocessor.joblib"]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = model_file.replace(".pkl", "").replace("_", " ").title()
    model = joblib.load(model_path)

    grouped = collapse_onehot_feature_names(feature_names)

    # === MultiOutput models – per-target plots
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        print(f" Multi-target model detected: {model_name}")
        for i, est in enumerate(model.estimators_):
            target = target_names[i] if i < len(target_names) else f"target_{i}"
            fi = est.feature_importances_
            aggr = {
                k: fi[[np.where(feature_names == f)[0][0] for f in v]].sum()
                for k, v in grouped.items()
            }
            total = sum(aggr.values())
            if total == 0:
                print(f"Skipping {model_name} – {target}: zero importances")
                continue
            aggr = {k: v / total for k, v in aggr.items()}

            importance_df = pd.DataFrame.from_dict(aggr, orient="index", columns=["Importance"])
            importance_df = importance_df.sort_values("Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            importance_df.plot(kind="barh", legend=False, color="steelblue")
            plt.xlabel("Relative Importance (Sum = 1)")
            plt.ylabel("Feature")
            plt.title(f"Top Features – {model_name} – {target}")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            output_path = f"../figures/feature_importance/feature_importance_{model_file.replace('.pkl', '')}_{target}.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f" Saved: {output_path}")
        continue

    # === XGBoost ===
    elif "xgboost" in model_name.lower() and hasattr(model, "get_booster"):
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type="gain")
        mean_importances = np.array([score_dict.get(f"f{i}", 0) for i in range(len(feature_names))])

    # === Tree-based models ===
    elif hasattr(model, "feature_importances_"):
        mean_importances = model.feature_importances_

    # === Linear models ===
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        mean_importances = coefs if coefs.ndim == 1 else np.mean(coefs, axis=0)

    else:
        print(f"Skipping {model_name}: No usable feature importances")
        continue

    # === Aggregate importances
    aggr_importances = {
        k: mean_importances[[np.where(feature_names == f)[0][0] for f in v]].sum()
        for k, v in grouped.items()
    }

    # === Normalize
    total_importance = sum(aggr_importances.values())
    if total_importance == 0:
        print(f"Skipping {model_name}: All-zero feature importances")
        continue
    aggr_importances = {k: v / total_importance for k, v in aggr_importances.items()}

    # === Plot
    importance_df = pd.DataFrame.from_dict(aggr_importances, orient="index", columns=["Importance"])
    importance_df = importance_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    importance_df.plot(kind="barh", legend=False, color="steelblue")
    plt.xlabel("Relative Importance (Sum = 1)")
    plt.ylabel("Feature")
    plt.title(f"Top Features – {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = f"../figures/feature_importance/feature_importance_{model_file.replace('.pkl', '')}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved: {output_path}")

