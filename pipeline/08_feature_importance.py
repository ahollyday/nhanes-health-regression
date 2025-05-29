import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re
from collections import defaultdict

# === Load preprocessor ===
preprocessor = joblib.load("../models/preprocessor.joblib")
feature_names = preprocessor.get_feature_names_out()

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

    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        # === Get feature importances ===
        all_importances = np.array([est.feature_importances_ for est in model.estimators_])
        mean_importances = np.mean(all_importances, axis=0)

        # === Aggregate importances ===
        grouped = collapse_onehot_feature_names(feature_names)
        aggr_importances = {
            k: mean_importances[[np.where(feature_names == f)[0][0] for f in v]].sum()
            for k, v in grouped.items()
        }

        # === Normalize to get relative importance
        total_importance = sum(aggr_importances.values())
        aggr_importances = {k: v / total_importance for k, v in aggr_importances.items()}

        # === Plot ===
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
        print(f"✅ Saved normalized feature importance plot to {output_path}")

