import os
import pandas as pd
import pyreadstat
import yaml
from functools import reduce

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
    'race_ethnicity': [7, 9],
    'has_health_insurance': [7, 9],
    'daily_sedentary_activity': [7777, 9999],
    'daily_intense_work': [7777, 9999],
    'weekly_hrs_worked': [77777, 99999]
}

def load_feature_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)["features"]

def apply_value_maps(df, features):
    for f in features:
        col = f["name"]
        if "map" in f and col in df.columns:
            df[col] = df[col].map(f["map"])
    return df

def remove_extrema_columnwise(df, col_name, lower=0.05, upper=0.95):
    q_low = df[col_name].quantile(lower)
    q_high = df[col_name].quantile(upper)
    original_len = len(df)
    df = df[(df[col_name] >= q_low) & (df[col_name] <= q_high)]
    removed = original_len - len(df)
    print(f"📉 Removed {removed} rows from '{col_name}' outside {int(lower*100)}th–{int(upper*100)}th percentiles")
    return df

def load_and_clean_nhanes_predict(year_suffix="H", output_filename="clean_data_2017_2018.csv"):
    print(f"📅 Loading NHANES prediction data for suffix: {year_suffix}")
    config_path = "../../data/config/features.yaml"
    features = load_feature_config(config_path)
    file_list = sorted(set(f["file"] for f in features))
    print(f"📆 Files to load: {file_list}")

    data_dir = "../../data/predict_raw"
    dfs = []

    for file_code in file_list:
        filename = f"{file_code.replace('_I', f'_{year_suffix}')}.xpt.txt"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue

        df, _ = pyreadstat.read_xport(path)
        print(f"✅ Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

        cols_for_file = [f for f in features if f["file"] == file_code]
        col_names = ["SEQN"] + [f["source"] for f in cols_for_file if f["source"] in df.columns]
        df = df[col_names].copy()

        rename_map = {f["source"]: f["name"] for f in cols_for_file}
        df.rename(columns=rename_map, inplace=True)

        for f in cols_for_file:
            col = f["name"]
            if col in df.columns and col in INVALID_VALUES:
                df[col] = df[col].astype("Float64")
                df[col] = df[col].replace(INVALID_VALUES[col], pd.NA)

        for f in cols_for_file:
            if f.get("drop_extrema", False) and f["name"] in df.columns:
                df = remove_extrema_columnwise(df, f["name"])

        df = apply_value_maps(df, cols_for_file)

        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), dfs)
    print(f"🔗 Merged dataframe shape: {merged.shape}")

    target_cols = [f["name"] for f in features if f["role"] == "target"]
    before = len(merged)
    merged = merged.dropna(subset=target_cols)
    after = len(merged)
    print(f"🛋 Dropped {before - after} rows with missing target values: {target_cols}")

    output_path = f"../../data/predict_processed/{output_filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned prediction data to {output_path}")

    # Optional summary
    categorical = [f["name"] for f in features if f["type"] == "categorical"]
    numerical = [f["name"] for f in features if f["type"] in ["numeric", "numerical"]]
    print("\n🔎 Summary of cleaned prediction dataset:")
    print(f"Categorical features: {categorical}")
    print(f"Numerical features: {numerical}")
    print(f"Target variables: {target_cols}")
    print(f"Cleaned data shape: {merged.shape}")

    return merged

if __name__ == "__main__":
    load_and_clean_nhanes_predict()

