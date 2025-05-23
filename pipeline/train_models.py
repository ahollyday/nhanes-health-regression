import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import yaml


def load_feature_config(config_path='data/config/features.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_models():
    os.makedirs('models', exist_ok=True)

    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')

    # Load feature types
    features = load_feature_config()
    categorical = [f['name'] for f in features if f['type'] == 'categorical' and f['name'] in X_train.columns]
    numeric = [f['name'] for f in features if f['type'] == 'numeric' and f['name'] in X_train.columns]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    # Define models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'Ridge': Ridge(),
        'SVR': SVR()
    }

    # Hyperparameter grids (abbreviated)
    param_grids = {
        'RandomForest': {
            'regressor__estimator__n_estimators': [100],
            'regressor__estimator__max_depth': [5, 10]
        },
        'GradientBoosting': {
            'regressor__estimator__n_estimators': [100],
            'regressor__estimator__learning_rate': [0.1, 0.01]
        },
        'KNN': {
            'regressor__estimator__n_neighbors': [5, 9]
        },
        'Ridge': {
            'regressor__estimator__alpha': [0.1, 1.0, 10.0]
        },
        'SVR': {
            'regressor__estimator__C': [0.1, 1.0]
        }
    }

    # Train and save best model for each
    for name, estimator in models.items():
        print(f"\n🔧 Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(estimator))
        ])

        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        joblib.dump(grid.best_estimator_, f'models/{name}_best_model.joblib')
        print(f"✅ Saved: models/{name}_best_model.joblib")


if __name__ == '__main__':
    train_models()

