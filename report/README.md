{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb85d267-cab6-4062-bf30-a24263ae2278",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


# NHANES Health Regression Project

This project builds and evaluates predictive models using NHANES survey and lab data (2015–2016 and 2017–2018 cycles). The goal is to predict multiple health biomarkers from demographic, dietary, activity, and interview features. The project is structured around a modular and reproducible machine learning pipeline.

---

## 📌 Objective

To evaluate how well health-related features such as diet, activity, and demographics can predict key biomarkers like:
- **HDL cholesterol**
- **Glycohemoglobin (A1C)**
- **Triglycerides**

---

## 🔄 Pipeline Overview

The pipeline is broken into modular steps to ensure reproducibility:

1. **Data Loading & Cleaning**  
   (`01_load_clean.py`)  
   Merges NHANES datasets, standardizes variable names, and cleans extreme or missing values.

2. **Data Summarization**  
   (`02_summarize_data.py`)  
   Outputs basic distributions and summary statistics.

3. **Preprocessing**  
   (`03_preprocess_model_data.py`)  
   Encodes categorical features, imputes missing values, and standardizes numerical features using a `ColumnTransformer` saved to disk.

4. **Model Training**  
   - `04_train_models_GridSearchCV.py`  
   - `04_train_models_RandomizedSearchCV.py`  
   - `04_train_models_optuna.py`  
   Trains and tunes several models (RF, GBR, XGBoost, SVR, Ridge, KNN) using 10-fold CV. Uses Optuna for Bayesian optimization.

5. **Hyperparameter Visualization**  
   (`05_vis_hp_results.py`)  
   Plots optimization history and parameter importance.

6. **Testing and Prediction**  
   (`06_test_models.py`, `predict/`)  
   Applies final models to the test set and generates predicted values.

7. **Evaluation and Plotting**  
   (`07_evaluate_models.py`, `08_feature_importance.py`)  
   Generates:
   - Performance metrics (R², RMSE)
   - True vs. predicted scatterplots
   - Residual plots
   - Feature importance plots

---

## 📊 EDA Summary

Exploratory analysis was conducted on both features and targets. Visualizations include:
- Histograms of all numerical features and targets
- Correlation heatmap of all variables

EDA figures can be found in:
