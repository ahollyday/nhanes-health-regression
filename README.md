# NHANES Health Regression Project

This project uses NHANES 2015–2018 data to train predictive models that estimate health biomarkers based on demographic, lifestyle, and lab features.

---

## 📌 Objective

To evaluate whether features like age, sex, diet, activity, and lab measurements can predict key health targets:

- **HDL cholesterol**
- **Glycohemoglobin (A1C)**
- **Triglycerides**

---

## 🔄 Pipeline Structure

Each step of the ML pipeline is modular and reproducible:

1. **Load & Clean Data**  
   `01_load_clean.py` — Merges datasets and removes missing/extreme values.

2. **Summarize Dataset**  
   `02_summarize_data.py` — Outputs descriptive statistics and histograms.

3. **Preprocessing**  
   `03_preprocess_model_data.py` — Uses a `ColumnTransformer` to:
   - Encode categorical variables
   - Scale numerical features
   - Impute missing values

4. **Model Training**  
   - `04_train_models_GridSearchCV.py`  
   - `04_train_models_RandomizedSearchCV.py`  
   - `04_train_models_optuna.py`  
   Trains multiple regressors using CV and tuning:
   - Random Forest, Gradient Boosting, XGBoost, Ridge, SVR, KNN

5. **Hyperparameter Visualization**  
   `05_vis_hp_results.py` — Visualizes Optuna search progress and parameter influence.

6. **Testing & Prediction**  
   `06_test_models.py`, `predict/` — Applies tuned models to test data.

7. **Evaluation & Reporting**  
   `07_evaluate_models.py`, `08_feature_importance.py` — Generates:
   - R², RMSE scores
   - True vs. predicted plots
   - Residual plots
   - Feature importance charts

---

## 📊 EDA Summary

Exploratory plots include:
- Distribution histograms for all features and targets
- Correlation heatmap to guide feature selection

All EDA outputs are saved to `/figures/eda/`.

---

## ⚙️ Configuration

Model features and targets are defined in `config/features.yaml`. Targets can be updated or log-transformed here. The pipeline uses YAML-based configs for flexibility.

---

## 📁 Directory Structure


