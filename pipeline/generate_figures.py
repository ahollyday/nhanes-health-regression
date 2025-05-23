import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")


def generate_model_metrics_plot():
    df = pd.read_csv("summaries/metrics.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="R²", hue="Target")
    plt.title("Model R² by Target")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/model_r2_by_target.png")
    plt.close()


def generate_residual_plot():
    df = pd.read_csv("summaries/residuals.csv")
    g = sns.displot(
        data=df,
        x="Residual",
        col="Target",
        row="Model",
        kde=True,
        height=2.5,
        aspect=2,
        facet_kws={"margin_titles": True},
        color="steelblue"
    )
    g.set(xlim=(-6, 6))
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.tight_layout()
    g.savefig("figures/residual_distributions.png")
    plt.close()


def generate_feature_importance_plot():
    df = pd.read_csv("summaries/feature_importances.csv")
    df = df.groupby("Feature")["Importance"].mean().reset_index().sort_values(by="Importance", ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Importance", y="Feature")
    plt.title("Top 20 Most Important Features")
    plt.tight_layout()
    plt.savefig("figures/feature_importances.png")
    plt.close()


def main():
    generate_model_metrics_plot()
    generate_residual_plot()
    generate_feature_importance_plot()
    print("✅ All plots saved to figures/")


if __name__ == '__main__':
    main()

