import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import yaml


def load_feature_config(config_path='data/config/features.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_models():
    os.makedirs('summaries', exist_ok=True)

    # Load data and scaler
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test_scaled = pd.read_csv('data/processed/y_test.csv')
    scaler = joblib.load('data/processed/target_scaler.joblib')

    # Load target names
    features = load_feature_config()
    target_cols = [f['name'] for f in features if f.get('target')]

    metrics = []
    all_residuals = []

    for fname in os.listdir('models'):
        if not fname.endswith('_best_model.joblib'):
            continue

        model_name = fname.replace('_best_model.joblib', '')
        print(f"\n📊 Evaluating {model_name}...")

        model = joblib.load(f'models/{fname}')
        y_pred_scaled = model.predict(X_test)

        for i, col in enumerate(target_cols):
            # Invert scaling and log transform
            y_pred_logged = scaler.inverse_transform(y_pred_scaled)[:, i]
            y_true_logged = scaler.inverse_transform(y_test_scaled)[:, i]

            y_pred = np.exp(y_pred_logged)
            y_true = np.exp(y_true_logged)

            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            metrics.append({
                "Model": model_name,
                "Target": col,
                "R²": r2,
                "RMSE": rmse
            })

            all_residuals.append(pd.DataFrame({
                "Model": model_name,
                "Target": col,
                "True": y_true,
                "Predicted": y_pred,
                "Residual": y_true - y_pred
            }))

    # Save outputs
    metrics_df = pd.DataFrame(metrics)
    residuals_df = pd.concat(all_residuals, ignore_index=True)

    metrics_df.to_csv('summaries/metrics.csv', index=False)
    residuals_df.to_csv('summaries/residuals.csv', index=False)

    print("\n✅ Evaluation complete:")
    print("- summaries/metrics.csv")
    print("- summaries/residuals.csv")


if __name__ == '__main__':
    evaluate_models()

