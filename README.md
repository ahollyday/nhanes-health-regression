# NHANES Health Regression Pipeline

This project builds a reproducible machine learning pipeline to model health outcomes from the NHANES dataset using multi-output regression. It includes automated data cleaning, feature engineering, model training, evaluation, and residual diagnostics.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                     # Original NHANES .xpt files
│   ├── processed/               # Cleaned CSV + train/test arrays
│   ├── input/                   # Optional: custom inputs for prediction
│   └── config/
│       └── features.yaml        # Feature metadata + modeling flags
│
├── models/                      # Saved trained models (.pkl)
├── summaries/                   # Training/testing metrics & residuals
├── figures/
│   ├── eda/                     # Exploratory plots
│   └── evaluation/              # Final evaluation plots
│
├── pipeline/
│   ├── load_clean.py            # Merges + cleans NHANES raw data
│   ├── summarize_data.py        # Summary stats, histograms, correlation
│   ├── preprocess_model_data.py # Imputation, encoding, scaling, splitting
│   ├── train_models.py          # Trains & tunes ML models + residuals
│   ├── test_models.py           # Loads models, evaluates on test set
│   ├── evaluate_models.py       # Aggregates plots & CV effect
│   └── clean_project.py         # Safely deletes intermediate outputs
```

---

## 🚀 Pipeline Overview

Run each stage in order:

```bash
python load_clean.py
python summarize_data.py
python preprocess_model_data.py
python train_models.py
python test_models.py
python evaluate_models.py
```

---

## 🧼 Data Cleaning

- Feature selection via `features.yaml`
- Merging multiple NHANES .xpt files on `SEQN`
- Renaming columns
- Replacing special codes (e.g., 777, 999) with `NaN`
- Dropping extrema (if flagged)
- Dropping rows with missing targets

---

## 📊 Models Trained

Multi-output regressors:
- **Linear Regression (baseline)**
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Regression

---

## 📈 Outputs

- `train_metrics.csv` / `test_metrics.csv`: R² / RMSE by model and target
- `train_residuals.csv` / `test_residuals.csv`: Model residuals
- `train_test_metrics_scatter.png`: Compare train/test generalization
- `train_residuals_overlay.png`, `test_residuals_overlay.png`: Residual histograms
- `cv_fold_analysis.png`: R² / RMSE vs CV folds

---

## ⚙️ Feature Configuration

```yaml
- name: triglycerides
  source: LBXTR
  file: TRIGLY_I
  type: numeric
  role: target
  drop_extrema: true
  log_transform: true
  unit: mg/dL
```

Supports:
- `drop_extrema`
- `log_transform`
- `unit`
- `role: feature` or `target`

---

## 🧼 Clean Intermediate Files

```bash
python clean_project.py
```

---

## ✨ Author

Andrew Hollyday  
Ph.D. in Geophysics, Columbia University

MIT License