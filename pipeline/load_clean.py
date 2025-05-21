import pandas as pd
import numpy as np
import pyreadstat
import yaml
from pathlib import Path

def load_config(config_path="data/config/features.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_raw_xpt(file_path):
    df, _ = pyreadstat.read_xport(file_path)
    return df

def clean_nhanes(df, config):
    for feat in config["features"]:
        orig_col = feat["id"]
        new_col = feat["name"]

        if orig_col not in df.columns:
            continue

        df.rename(columns={orig_col: new_col}, inplace=True)

        if "missing_codes" in feat:
            for code in feat["missing_codes"]:
                df[new_col] = df[new_col].replace(code, np.nan)

        if feat["type"] == "categorical" and "categories" in feat:
            df[new_col] = df[new_col].map(feat["categories"])
            df[new_col] = pd.Categorical(df[new_col])

    return df

def run_clean_pipeline(xpt_paths, config_path="data/config/features.yaml"):
    config = load_config(config_path)

    # Load and merge all raw files
    dfs = [load_raw_xpt(path) for path in xpt_paths]
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on="SEQN", how="outer")

    # Clean
    df_cleaned = clean_nhanes(df_merged, config)

    return df_cleaned

if __name__ == "__main__":
    raw_dir = Path("data/raw/")
    xpt_paths = list(raw_dir.glob("*.xpt.txt"))
    df_cleaned = run_clean_pipeline(xpt_paths)
    df_cleaned.to_csv("data/processed/cleaned_nhanes.csv", index=False)
    print("✅ Cleaned data saved to data/processed/cleaned_nhanes.csv")

