import os
import pandas as pd
import pyreadstat
import yaml
from functools import reduce

# Define per-column invalid NHANES codes (after renaming)
INVALID_VALUES = {
    'drinks_per_day': [777, 999],
    'alc_freq_past12mo': [777, 999],
    'intense_activity_work': [7, 9],
    'some_activity_work': [7, 9],
    'monthly_family_income': [77, 99],
    'has_20k_savings': [7, 9],
    'total_savings': [77, 99],
    'ever_smoked_100': [7, 9],
    'sex': [7, 9],
    'race_ethnicity': [7, 9]
}

def load_feature_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)["features"]

def remove_min_max_extrema(df, features):
    drop_cols = [f["name"] for f in features if f.get("drop_extrema", False)]
    for col in drop_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            before = len(df)
            df = df[(df[col] != min_val) & (df[col] != max_val)]
            after = len(df)
            print(f"📉 Removed {before - after} rows from '{col}' equal to min ({min_val}) or max ({max_val})")
    return df

def apply_value_maps(df, features):
    for f in features:
        col = f["name"]
        if "map" in f and col in df.columns:
            df[col] = df[col].map(f["map"])
    return df

def load_and_clean_nhanes():
    print("📅 Loading and merging NHANES files...")
    config_path = "../data/config/features.yaml"
    features = load_feature_config(config_path)
    file_list = sorted(set(f["file"] for f in features))
    print(f"📆 Files to load: {file_list}")

    data_dir = "../data/raw"
    dfs = []
    for file_code in file_list:
        path = os.path.join(data_dir, f"{file_code}.xpt.txt")
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
        df, _ = pyreadstat.read_xport(path)
        print(f"✅ Loaded {file_code}: {df.shape[0]} rows, {df.shape[1]} columns")
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), dfs)
    print(f"🔗 Merged dataframe shape: {merged.shape}")

    rename_map = {f["source"]: f["name"] for f in features}
    keep_cols = ["SEQN"] + list(rename_map.keys())
    try:
        subset = merged[keep_cols].copy()
    except KeyError as e:
        missing = [col for col in keep_cols if col not in merged.columns]
        print(f"❌ Missing columns in merged data: {missing}")
        raise

    df_clean = subset.rename(columns=rename_map)
    print(f"🪜 Renamed columns: {list(df_clean.columns)}")

    for col, bad_values in INVALID_VALUES.items():
        if col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype("Float64")
                df_clean[col] = df_clean[col].replace(bad_values, pd.NA)
            except Exception as e:
                print(f"⚠️ Skipping replacement for column {col} due to error: {e}")

    df_clean = remove_min_max_extrema(df_clean, features)
    df_clean = apply_value_maps(df_clean, features)

    # Drop rows where any target is missing (required for multi-output regression)
    target_cols = [f["name"] for f in features if f["role"] == "target"]
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=target_cols)
    after = len(df_clean)
    print(f"🛋 Dropped {before - after} rows with missing target values: {target_cols}")

    output_path = "../data/processed/clean_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned data to {output_path}")

if __name__ == "__main__":
    load_and_clean_nhanes()

