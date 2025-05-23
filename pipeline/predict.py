import pandas as pd
import numpy as np
import joblib
import yaml
import os

from sklearn.preprocessing import StandardScaler


def load_feature_config(config_path='data/config/features.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_best_model(metrics_path='summaries/metrics.csv'):
    df = pd.read_csv(metrics_path)
    avg_r2 = df.groupby("Model")["R²"].mean()
    best_model = avg_r2.idxmax()
    return f"models/{best_model}_best_model.joblib"


def predict(input_path='data/input/input.csv', output_path='predictions/predictions.csv'):
    os.makedirs('predictions', exist_ok=True)

    # Load input
    X_new = pd.read_csv(input_path)

    # Load best model
    model_path = get_best_model()
    model = joblib.load(model_path)

    # Load scaler and config
    scaler = joblib.load('data/processed/target_scaler.joblib')
    features = load_feature_config()
    target_cols = [f['name'] for f in features if f.get('target')]

    # Predict (model includes preprocessing pipeline)
    y_pred_scaled = model.predict(X_new)

    # Invert scaling and log transform
    y_pred_logged = scaler.inverse_transform(y_pred_scaled)
    y_pred_original = np.exp(y_pred_logged)

    # Wrap in DataFrame and save
    df_pred = pd.DataFrame(y_pred_original, columns=target_cols, index=X_new.index)
    df_pred.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")


if __name__ == '__main__':
    predict()

