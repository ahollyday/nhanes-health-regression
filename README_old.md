# NHANES Machine Learning Pipeline

This project builds a reproducible machine learning pipeline to model and predict health outcomes using the NHANES dataset. It includes scripts for cleaning, preprocessing, training, evaluation, and visualization.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                     # NHANES XPT data files
│   ├── processed/               # Cleaned + preprocessed datasets
│   └── input/                   # New rows for prediction
│
├── models/                     # Saved trained models
├── summaries/                  # Metrics, residuals, feature importances
├── figures/                    # Saved plots for reporting
├── predictions/                # Predictions for new input
├── data/config/features.yaml   # Feature definitions and metadata
│
├── load_clean.py               # Merges and cleans raw NHANES data
├── summarize_data.py           # Computes missingness and distributions
├── preprocess_model_data.py    # Scales + splits train/test datasets
├── train_models.py             # Trains and tunes regression models
├── evaluate_models.py          # Computes R^2 and RMSE, outputs residuals
├── feature_importance.py       # Extracts feature importances (tree models)
├── generate_figures.py         # Produces key summary plots
├── predict.py                  # Makes predictions on new data
├── run_pipeline.py             # Runs full end-to-end pipeline
└── README.md                   # Project documentation
```

---

## 🔁 Pipeline Workflow

Run everything with:
```bash
python run_pipeline.py
```

Or step-by-step:
```bash
python load_clean.py
python summarize_data.py
python preprocess_model_data.py
python train_models.py
python evaluate_models.py
python feature_importance.py
python generate_figures.py
```

---

## 🧠 Models Trained
- **Random Forest**
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**
- **Ridge Regression**
- **Support Vector Regression (SVR)**

All models use a `Pipeline` with preprocessing (`StandardScaler`, `OneHotEncoder`) and multi-output regressors.

---

## 📊 Outputs
- `summaries/metrics.csv`: R² and RMSE per model and target
- `summaries/residuals.csv`: residuals per prediction
- `summaries/feature_importances.csv`: mean importance across targets
- `figures/`: plots of model performance and residuals
- `predictions/predictions.csv`: outputs from `predict.py`

---

## ✨ Example Usage

Predict on new data:
```bash
python predict.py
```

Edit `data/input/input.csv` to provide rows for prediction.

---

## ⚙️ Configurable Features
Defined in `data/config/features.yaml`, you can:
- Mark variables as `numeric`, `categorical`, or `target`
- Easily toggle which features and targets are used

---


---

## ⚙️ `load_clean.py` – Data Cleaning & Preparation

This script reads, cleans, and merges NHANES data according to rules defined in `features.yaml`, then outputs a cleaned CSV ready for modeling.

### 🧼 Key Steps Performed:

1. **Load selected features** from `features.yaml`
2. **Merge multiple `.xpt.txt` files** on `SEQN`
3. **Rename columns** to friendly names (e.g., `BMXBMI` → `bmi`)
4. **Replace special missing values** (e.g., 7, 9, 777, 999) with `pd.NA`
5. **Remove extrema** for numeric features flagged with `drop_extrema: true`
6. **Map categorical values** to human-readable labels (e.g., `1` → `Male`)
7. **Drop rows with missing target values** to ensure valid multi-output regression input
8. **Save final DataFrame** to `../data/processed/clean_data.csv`

---

## 📊 `summarize_data.py` – Exploratory Summary

This script provides a quick overview of the cleaned dataset:

- Number of rows and columns
- Count and percentage of missing values per column
- Summary statistics for numeric variables
- Value counts for categorical variables

Useful for sanity checks before modeling.

---

## 🔧 Configuration via `features.yaml`

All variables used in the pipeline are defined in `features.yaml` and include:

- `name`: target name for modeling
- `source`: original NHANES code
- `file`: raw data file
- `type`: `numeric` or `categorical`
- `role`: `feature` or `target`
- `drop_extrema`: optional, for numeric range cleaning
- `map`: optional, dictionary to map codes to labels

Example:

```yaml
- name: sex
  source: RIAGENDR
  file: DEMO_I
  type: categorical
  role: feature
  map:
    1: "Male"
    2: "Female"

---

## 📌 Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, seaborn, matplotlib, joblib, pyyaml

---

## ✅ Next Steps
- Add Tableau dashboards using `summaries/` and `figures/`
- Extend to other NHANES cycles
- Package into a lightweight API or Streamlit app

---

## Author
[Your Name Here]

MIT License

