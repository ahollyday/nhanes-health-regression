import os
import joblib
import pandas as pd
import numpy as np
import yaml


def load_feature_config(config_path='data/config/features.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_feature_importance():
    features = load_feature_config()
    cat_features = [f['name'] for f in features if f['type'] == 'categorical']
    num_features = [f['name'] for f in features if f['type'] == 'numeric']
    all_features = cat_features + num_features

    importances = []

    for fname in os.listdir('models'):
        if not fname.endswith('_best_model.joblib'):
            continue

        model_name = fname.replace('_best_model.joblib', '')
        model = joblib.load(f'models/{fname}')
        regressor = model.named_steps['regressor']

        if hasattr(regressor, 'estimators_'):
            for i, est in enumerate(regressor.estimators_):
                if hasattr(est, 'feature_importances_'):
                    fi = est.feature_importances_
                    df = pd.DataFrame({
                        'Model': model_name,
                        'Target': i,
                        'Feature': model.named_steps['preprocessor'].get_feature_names_out(),
                        'Importance': fi
                    })
                    importances.append(df)

    final_df = pd.concat(importances, ignore_index=True)
    os.makedirs('summaries', exist_ok=True)
    final_df.to_csv('summaries/feature_importances.csv', index=False)
    print("✅ Feature importances saved to summaries/feature_importances.csv")


if __name__ == '__main__':
    extract_feature_importance()

