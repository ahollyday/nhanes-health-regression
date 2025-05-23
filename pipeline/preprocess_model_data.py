import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from joblib import dump

def load_feature_config(path="../data/config/features.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)["features"]

def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    return np.expm1(x)

def preprocess():
    print("📥 Loading cleaned data...")
    df = pd.read_csv("../data/processed/clean_data.csv")
    features = load_feature_config()

    numeric_features = [f["name"] for f in features if f["type"] == "numeric" and f["role"] == "feature"]
    categorical_features = [f["name"] for f in features if f["type"] == "categorical" and f["role"] == "feature"]
    target_features = [f["name"] for f in features if f["role"] == "target"]

    X = df[numeric_features + categorical_features].copy()
    y = df[target_features].copy()

    # Log-transform targets
    y_log = np.log1p(y)

    print("🧪 Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    print("🧼 Building preprocessing pipeline...")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("⚙️ Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("💾 Saving preprocessed data and pipeline...")
    os.makedirs("../data/processed", exist_ok=True)
    dump(preprocessor, "../models/preprocessor.joblib")
    np.savez("../data/processed/train_test_data.npz",
             X_train=X_train_transformed, X_test=X_test_transformed,
             y_train=y_train.values, y_test=y_test.values,
             feature_names=preprocessor.get_feature_names_out())
    print("✅ Done.")

if __name__ == "__main__":
    preprocess()

