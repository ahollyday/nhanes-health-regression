# NHANES Health Regression Project

This project uses NHANES 2015–2016 data to train predictive models that estimate health biomarkers based on demographic, lifestyle, and lab features.

---

## Objective

This pipeline is designed as a flexible modeling tool for predicting a wide range of clinical, behavioral, and biomarker variables available in NHANES. Any numerical target defined in the config file can be modeled, allowing users to explore relationships between diverse health indicators and accessible features like diet, activity, and demographics.

While some targets (e.g., total cholesterol) require closely related biomarkers (e.g., HDL, apolipoprotein) for high predictive accuracy, others can be reasonably estimated from indirect features. This makes the tool adaptable across use cases—from exploratory modeling to targeted prediction when partial lab panels are available.

A separate example report demonstrates the pipeline's application on one target, including all evaluation outputs.

---

## Pipeline Structure

Each step of the ML pipeline is modular and reproducible:

1. **Load & Clean Data**  
   `01_load_clean.py` — Merges datasets and removes missing values. Applies 5th–95th percentile filtering to features flagged with `drop_extrema: true` to reduce outlier skew.

2. **Summarize Dataset**  
   `02_summarize_data.py` — Outputs descriptive statistics and histograms.

3. **Preprocessing**  
   `03_preprocess_model_data.py` — Uses a `ColumnTransformer` to:
   - Log transform targets when specified
   - Encode categorical variables
   - Scale numerical features
   - Impute missing values

4. **Model Training**
   - `04_train_models_optuna.py`  
   - `04_train_models_GridSearchCV.py` (not yet complete)
   - `04_train_models_RandomizedSearchCV.py` (not yet compelete)


   Trains multiple regressors using CV and tuning:
   - Random Forest, Gradient Boosting, XGBoost, SVR, KNN
   - A baseline MultiOutput Linear Regression model is also included for comparison

   These models span tree-based, linear, kernel-based, and distance-based techniques, offering a balance between flexibility and interpretability.

   #### Model Types

   The pipeline supports a diverse set of regression models, including:

   - **Tree-based models**:  
     - *Random Forest*: robust to outliers, good for non-linear relationships  
     - *Gradient Boosting* and *XGBoost*: strong predictive performance, effective for structured/tabular data

   - **Linear models**:  
     - *Linear Regression (baseline)*: provides a simple benchmark; fast and interpretable

   - **Kernel-based models**:  
     - *Support Vector Regression (SVR)*: flexible but sensitive to parameter tuning

   - **Distance-based models**:  
     - *K-Nearest Neighbors (KNN)*: intuitive, performs well with localized patterns but may struggle with noisy features

   These choices balance interpretability, non-linearity, and robustness, and can be tuned using either grid search, randomized search, or Optuna-based (recommended) optimization.

6. **Hyperparameter Visualization**  
   `05_vis_hp_results.py` — Visualizes Optuna search progress and parameter influence:
   - Optimization history
   - Parallel coordinate plots
   - Relative hyperparameter importance

7. **Testing & Prediction**  
   `06_test_models.py`, `predict/` — Applies tuned models to unseen data and saves predicted outputs.

8. **Evaluation & Reporting**  
   `07_evaluate_models.py`, `08_feature_importance.py` — Generates:
   - R², RMSE scores (train/test)
   - Residual error distributions
   - Feature importance charts

---

## Generalization to New NHANES Cycles

The `predict/` subdirectory allows trained models to be applied to earlier NHANES data cycles (e.g., 2013–2014). Using the same feature definitions from `features.yaml`, these scripts:

- Load and clean a new dataset: `01_load_clean_pred.py`
- Summarize data: `02_summarize_data_pred.py`
- Preprocess: `03_preprocess_pred.py`
- Predict: `04_predict.py`

These can be executed individually or together via `run_pred_pipeline.bash`. Predictions and residuals are saved to `summaries/`, and visual outputs go to `figures/predictions/`.

This demonstrates the pipeline's flexibility to generalize across survey years.

---

## EDA & Diagnostics

Exploratory plots and diagnostics help inform model configuration and feature selection:

- Distribution histograms for all features and targets
- Correlation heatmap to guide feature selection
- Univariate R² plots for each feature vs. target
- Residual histograms and error distributions
- Feature importance charts (per model and target)
- True vs. predicted scatterplots for each model

All EDA and diagnostic outputs are saved to `/figures/`.

---

## Configuration

Model features and targets are defined in `data/config/features.yaml`. Targets must be numerical. To apply a log transform, set `log_transform: true` under the target's config. *(Note: `log_transform` should only be used for targets—not features.)*

Only numerical targets are supported. 

Features with `drop_extrema: true` will be filtered to exclude values below the 5th and above the 95th percentiles.

Cross-validation folds and Optuna run time limits can be modified in the training scripts. For example, `n_trials`, `cv`, and `timeout` can be specified in `04_train_models_optuna.py`.

---

## How to Run

1. **Install requirements**  
   Install dependencies using `pip install -r requirements.txt` or with conda `conda install --file requirements.txt`. Required packages include:

   - pandas, scikit-learn, matplotlib, seaborn, joblib, optuna

3. **Download NHANES Data**  
   Place raw `.xpt.txt` files in the appropriate `data/raw/` and `data/predict_raw/` directories.

4. **Configure features**  
   Edit `data/config/features.yaml` to define variable roles, units, transforms, and thresholds.

5. **Run training pipeline**  
   From the root directory, execute:
   ```bash
   bash run_pipeline.bash
   ```
   This will clean, preprocess, train, evaluate, and generate all outputs from `data/raw/`.

6. **Run prediction on a new cycle**  
   To apply models to new survey data (e.g., 2013–2014), run:
   ```bash
   bash predict/run_pred_pipeline.bash
   ```
   This uses `predict_raw/` and writes predictions to `summaries/` and `figures/predictions/`.

---

## Directory Structure

```
nhanes-health-regression/
├── data/
│   ├── raw/                  # Raw NHANES data files 2015–2016 (.xpt.txt)
│   ├── processed/            # Cleaned data and train/test splits (train)
│   ├── predict_processed/    # Cleaned data and train/test splits (pred)
│   ├── predict_raw/          # Raw NHANES data files 2013–2014 (.xpt.txt)
│   └── config/               # Feature config files
│       └── features.yaml
│ 
├── figures/
│   ├── eda/                  # Correlation plots, histograms
│   │   ├── boxplot_?.png     # Boxplots for cat. vars by target
│   │   ├── corr_heat.png     # Linear correlations
│   │   ├── hist_feat.png     # Feature distributions
│   │   ├── hist_targ.png     # Target distributions
│   │   └── univar_?.png      # R² of single-feature regressions
│   │
│   ├── evaluation/           # Model metrics, residuals, CV analysis
│   │   └── optuna_visuals    # Hyperparameter tuning results (by model)
│   │       ├── opt_hist.png  # Optimization history (R² over trials)
│   │       ├── p_coord.png   # Parallel plot of sampled parameters
│   │       └── p_imp.png     # Bar chart of hyperparameter importances
│   │
│   ├── feature_importance    # Feature importances by target
│   └── predictions           # EDA and true vs predicted 
│       └── eda
│ 
├── models/                   # Serialized trained models (.pkl)
├── predict/                  # Scripts for applying models to new cycles
│   ├── 01_load_clean_pred.py
│   ├── 02_summarize_data_pred.py
│   ├── 03_preprocess_pred.py
│   ├── 04_predict.py
│   └── run_pred_pipeline.bash
│
├── summaries/                # R², RMSE scores and residuals per model
│
├── pipeline/                 # All scripts (01–08) for end-to-end pipeline
│   ├── 01_load_clean.py
│   ├── 02_summarize_data.py
│   ├── 03_preprocess_model_data.py
│   ├── 04_train_models_optuna.py
│   ├── 05_vis_hp_results.py
│   ├── 06_test_models.py
│   ├── 07_evaluate_models.py
│   └── 08_feature_importance.py
│
└── README.md                 # This file
```

## Run with Docker

Build and run the full pipeline in a reproducible containerized environment:

```bash
# Build the image
docker build -t nhanes-regression .

# Run the pipeline
docker run -it nhanes-regression

# (Optional) Save outputs locally
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/summaries:/app/summaries \
  -v $(pwd)/figures:/app/figures \
  -v $(pwd)/optuna_studies:/app/optuna_studies \
  nhanes-regression


