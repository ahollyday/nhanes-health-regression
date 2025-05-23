import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

    print("\n📉 Generating correlation heatmap...")
    corr = df.drop(columns='SEQN', errors='ignore').corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
        ax=ax
    )
    plt.title("Linear Correlation Matrix", fontsize=12)

    # Keep tick labels but remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Whiten specified axis labels
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    if xticklabels:
        xticklabels[-1].set_color("white")
    if yticklabels:
        yticklabels[0].set_color("white")

    # Bold target variable labels
    for tick in xticklabels:
        if tick.get_text() in target_features:
            tick.set_fontweight("bold")
    for tick in yticklabels:
        if tick.get_text() in target_features:
            tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("../figures/eda/correlation_heatmap.png", dpi=300)
    plt.close()
    print("✅ Saved correlation heatmap to ../figures/eda/correlation_heatmap.png")

    print("\n📊 Generating histograms for numeric features...")
    os.makedirs("../figures/eda", exist_ok=True)
    n = len(numeric_features)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4), dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_xlabel(f"{col} ({units.get(col, '')})")
            axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("../figures/eda/hist_numeric_features.png", dpi=300)
    plt.close()
    print("✅ Saved numeric feature histograms to ../figures/eda/hist_numeric_features.png")

    print("\n📊 Generating histograms for targets...")
    n = len(target_features)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n * 6, 4), dpi=300)
    axes = axes if n > 1 else [axes]
    for i, col in enumerate(target_features):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_xlabel(f"{col} ({units.get(col, '')})")
            axes[i].set_ylabel("Count")
    fig.suptitle("Target Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("../figures/eda/hist_targets.png", dpi=300)
    plt.close()
    print("✅ Saved target histograms to ../figures/eda/hist_targets.png")

    print("\n📉 Generating univariate linear regression plots...")
    r2_table = pd.DataFrame(index=numeric_features, columns=target_features, dtype=float)
    n_cols = 3
    n_rows = math.ceil(len(numeric_features) / n_cols)
    target_colors = dict(zip(target_features, sns.color_palette("tab10", len(target_features))))

    for target in target_features:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=300)
        axes = axes.flatten()
        for i, feature in enumerate(numeric_features):
            ax = axes[i]
            subset = df[[feature, target]].dropna()
            X = subset[[feature]].values
            y = subset[target].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            r2_table.loc[feature, target] = r2

            sns.regplot(
                x=feature,
                y=target,
                data=df,
                ax=ax,
                ci=95,
                scatter_kws={'alpha': 1, 's': 8, 'color': target_colors[target]},
                line_kws={'color': 'red'}
            )

            ax.set_title(f"$R^2$ = {r2:.2f}")
            ax.set_xlabel(f"{feature} ({units.get(feature, '')})")
            ax.set_ylabel(f"{target} ({units.get(target, '')})")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Linear fit'),
            Line2D([0], [0], color='red', lw=2, alpha=0.3, label='95% CI')
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=8
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f"../figures/eda/univariate_{target}.png", dpi=300)
        plt.close()

    print("✅ Saved univariate regression plots to ../figures/eda/")

if __name__ == "__main__":
    summarize()

