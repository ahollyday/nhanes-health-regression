import pandas as pd
import numpy as np
import yaml
import os
from joblib import load, dump

def load_feature_config(path="../../data/config/features.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)["features"]

def preprocess_for_prediction(input_filename="clean_data_2017_2018.csv", output_filename="X_2017_2018.npy"):
    print(" Loading cleaned prediction data...")
    df = pd.read_csv(f"../../data/predict_processed/{input_filename}")
    features = load_feature_config()

    numeric_features = [f["name"] for f in features if f["type"] in ["numeric", "numerical"] and f["role"] == "feature"]
    categorical_features = [f["name"] for f in features if f["type"] == "categorical" and f["role"] == "feature"]
    target_features = [f["name"] for f in features if f["role"] == "target"]

    X = df[numeric_features + categorical_features].copy()
    y = df[target_features].copy()  # Optional: may not be used

    print(" Loading fitted preprocessor...")
    preprocessor = load("../../models/preprocessor.joblib")

    print(" Transforming features...")
    X_transformed = preprocessor.transform(X)

    print(" Saving transformed features...")
    os.makedirs("../../data/predict_processed", exist_ok=True)
    np.save(f"../../data/predict_processed/{output_filename}", X_transformed)

    print(" Done.")
    return X_transformed, df["SEQN"].values, y  # in case you want to pair predictions with IDs/targets

if __name__ == "__main__":
    preprocess_for_prediction()

