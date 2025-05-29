import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump

def load_feature_config(path="../data/config/features.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)["features"]

def preprocess():
    print("\U0001F4E5 Loading cleaned data...")
    df = pd.read_csv("../data/processed/clean_data.csv")
    features = load_feature_config()

    numeric_features = [f["name"] for f in features if f["type"] in ["numeric", "numerical"] and f["role"] == "feature"]
    categorical_features = [f["name"] for f in features if f["type"] == "categorical" and f["role"] == "feature"]
    target_features = [f["name"] for f in features if f["role"] == "target"]

    log_transform_targets = [f["name"] for f in features if f["role"] == "target" and f.get("log_transform", False)]

    X = df[numeric_features + categorical_features].copy()
    y = df[target_features].copy()
    y[log_transform_targets] = np.log1p(y[log_transform_targets])

    print("\U0001F9EA Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\U0001FA9C Building preprocessing pipeline...")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("\u2699\ufe0f Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("\U0001F4BE Saving preprocessed data and pipeline...")
    os.makedirs("../data/processed", exist_ok=True)
    dump(preprocessor, "../models/preprocessor.joblib")
    np.savez("../data/processed/train_test_data.npz",
             X_train=X_train_transformed, X_test=X_test_transformed,
             y_train=y_train.values, y_test=y_test.values,
             feature_names=preprocessor.get_feature_names_out())
    print("\u2705 Done.")

if __name__ == "__main__":
    preprocess()

