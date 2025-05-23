import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

def load_feature_config(path="../data/config/features.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)["features"]

def get_units_dict(features):
    return {f["name"]: f.get("unit", "") for f in features}

def summarize():
    print("📥 Loading cleaned data...")
    df = pd.read_csv("../data/processed/clean_data.csv")
    features = load_feature_config()

    numeric_features = [f["name"] for f in features if f["type"] == "numeric" and f["role"] == "feature"]
    categorical_features = [f["name"] for f in features if f["type"] == "categorical"]
    target_features = [f["name"] for f in features if f["role"] == "target"]
    units = get_units_dict(features)

    print(f"📂 Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n📊 Missing value summary:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n📉 Percentage of missing values per column:")
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct[missing_pct > 0])

    print("\n📈 Descriptive stats for numeric features:")
    print(df[numeric_features + target_features].describe().T)

    for col in categorical_features:
        if col in df.columns:
            print(f"\n🔠 Value counts for categorical feature: {col}")
            print(df[col].value_counts(dropna=False))

    print("\n📸 Generating histograms for numeric features...")
    os.makedirs("../figures/eda", exist_ok=True)
    n = len(numeric_features)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4), dpi=200)
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"{col} ({units.get(col, '')})")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig("../figures/eda/hist_numeric_features.png")
    plt.close()
    print("✅ Saved numeric feature histograms to ../figures/eda/hist_numeric_features.png")

    print("\n📊 Generating histograms for targets...")
    n = len(target_features)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n * 6, 4), dpi=200)
    axes = axes if n > 1 else [axes]
    for i, col in enumerate(target_features):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"{col} ({units.get(col, '')})")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
    plt.tight_layout()
    plt.savefig("../figures/eda/hist_targets.png")
    plt.close()
    print("✅ Saved target histograms to ../figures/eda/hist_targets.png")

    print("\n📊 Generating offset boxplots of targets by each categorical feature...")
    n_cols = 2
    n_rows = int(np.ceil(len(categorical_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), dpi=200)
    axes = axes.flatten()

    base_box_param = dict(
        whis=(5, 95), widths=0.18, patch_artist=True,
        flierprops=dict(marker='.', markeredgecolor='black', markersize=3, linestyle='none'),
        medianprops=dict(color='black', linewidth=1),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8)
    )

    offsets = np.linspace(-0.2, 0.2, len(target_features))
    colors = sns.color_palette("tab10", len(target_features))

    for i, cat in enumerate(categorical_features):
        ax = axes[i]
        categories = sorted(df[cat].dropna().unique())
        nb_groups = len(categories)

        for j, target in enumerate(target_features):
            data = [df.loc[(df[cat] == c) & df[target].notna(), target] for c in categories]
            valid_data = [d for d in data if len(d) > 0]
            valid_pos = [p + offsets[j] for d, p in zip(data, np.arange(nb_groups)) if len(d) > 0]
            valid_data = [d for d in data if len(d) > 0]
            if valid_data:
                ax.boxplot(valid_data, positions=valid_pos,
                           boxprops=dict(facecolor=colors[j]), **base_box_param)

        ax.set_xticks(np.arange(nb_groups))
        ax.set_xticklabels(categories, rotation=30, ha='right')
        ax.set_xlabel(cat)
        ax.set_ylabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("../figures/eda/box_targets_by_categoricals.png")
    plt.close()
    print("✅ Saved offset boxplot figure to ../figures/eda/box_targets_by_categoricals.png")

if __name__ == "__main__":
    summarize()

