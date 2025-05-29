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

def remove_min_max_extrema(df, features):
    drop_cols = [f["name"] for f in features if f.get("drop_extrema", False)]
    for col in drop_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df = df[(df[col] != min_val) & (df[col] != max_val)]
    return df

def apply_value_maps(df, features):
    for f in features:
        col = f["name"]
        if "map" in f and col in df.columns:
            df[col] = df[col].map(f["map"])
    return df

def load_and_clean_nhanes_predict(year_suffix="J", output_filename="clean_data_2017_2018.csv"):
    print(f"📅 Loading NHANES data for suffix: {year_suffix}")
    config_path = "../../data/config/features.yaml"
    features = load_feature_config(config_path)
    file_list = sorted(set(f["file"] for f in features))
    
    data_dir = f"../../data/predict_raw"
    dfs = []
    for file_code in file_list:
        file_with_suffix = f"{file_code.replace('_I', f'_{year_suffix}')}.xpt.txt"
        path = os.path.join(data_dir, file_with_suffix)
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
        df, _ = pyreadstat.read_xport(path)
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), dfs)
    rename_map = {f["source"]: f["name"] for f in features}
    keep_cols = ["SEQN"] + list(rename_map.keys())
    subset = merged[keep_cols].copy()
    df_clean = subset.rename(columns=rename_map)

    for col, bad_values in INVALID_VALUES.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("Float64")
            df_clean[col] = df_clean[col].replace(bad_values, pd.NA)

    df_clean = remove_min_max_extrema(df_clean, features)
    df_clean = apply_value_maps(df_clean, features)

    target_cols = [f["name"] for f in features if f["role"] == "target"]
    df_clean = df_clean.dropna(subset=target_cols)

    output_path = f"../../data/predict_processed/{output_filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned data to {output_path}")
    return df_clean

if __name__ == "__main__":
    # This can be modified or turned into CLI later
    load_and_clean_nhanes_predict()

