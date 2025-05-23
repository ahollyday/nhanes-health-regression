# NHANES Machine Learning Pipeline

This project builds a reproducible machine learning pipeline to model and predict health outcomes using the NHANES dataset. It supports end-to-end tasks: loading, cleaning, preprocessing, training, evaluation, and reporting.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                      # NHANES XPT data files
│   ├── processed/                # Cleaned + preprocessed datasets
│   ├── input/                    # New rows for prediction
│   └── config/
│       └── features.yaml         # Feature definitions and metadata
│
├── models/                       # Saved trained models
├── summaries/                    # Metrics, residuals, feature importances
├── figures/                      # Saved plots for reporting
├── predictions/                  # Predictions for new input
│
├── load_clean.py                # Merges, renames, cleans raw data
├── summarize_data.py            # Missingness + EDA summaries
├── preprocess_model_data.py     # Preprocessing and train/test split
├── train_models.py              # Model training and tuning
├── evaluate_models.py           # R², RMSE, residuals
├── feature_importance.py        # Tree-based importance extraction
├── generate_figures.py          # Summary plots and charts
├── predict.py                   # Run model on new input
├── run_pipeline.py              # Run full pipeline in sequence
└── README.md                    # Project documentation
```

---

## 🔁 Pipeline Workflow

Run the full pipeline:

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

## ⚙️ `load_clean.py` – Data Cleaning

This script prepares raw NHANES data for modeling. It performs:

1. Feature selection via `features.yaml`
2. Merge of multiple `.xpt.txt` files on `SEQN`
3. Renaming to human-readable feature names
4. Invalid value replacement (e.g., 7, 9, 777 → `pd.NA`)
5. Extrema filtering for numeric features
6. Categorical mapping (e.g., `1 → Male`)
7. Dropping rows with missing target values, required for multi-output models
8. CSV export to `data/processed/clean_data.csv`

---

## 📊 `summarize_data.py` – Data Exploration

This script provides:

- Missing value counts and percentages
- Descriptive stats for numeric features
- Value counts for categorical features

Ideal for pre-modeling checks and understanding data quality.

---

## 📄 `features.yaml` – Feature Configuration

Located in `data/config/`, this file controls:

- Feature names, types, and sources
- Role in modeling (`feature` or `target`)
- Optional extrema filtering (`drop_extrema: true`)
- Optional value mappings

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
```

---

## 📈 Models Supported

- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Ridge Regression
- Support Vector Regression (SVR)

All models use a `Pipeline` with preprocessing (scaling, encoding) and support multi-output regression (predicting multiple targets simultaneously).

---

## 📤 Outputs

- `summaries/metrics.csv` – R² and RMSE per model and target
- `summaries/residuals.csv` – prediction residuals
- `summaries/feature_importances.csv` – tree-based importances
- `figures/` – plots of model performance and residuals
- `predictions/predictions.csv` – predictions for new data

---

## 🧠 Example: Predict on New Data

Update `data/input/input.csv`, then run:

```bash
python predict.py
```

Predictions will be saved to `predictions/predictions.csv`.

---

## ✅ Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `scikit-learn`, `pyreadstat`, `pyyaml`, `matplotlib`, `seaborn`, `joblib`

Install via:

```bash
pip install -r requirements.txt
```

---

## 🧭 Next Steps

- Add Tableau dashboard from `summaries/` and `figures/`
- Extend to additional NHANES cycles
- Package into a Streamlit app or lightweight API

---

## 📇 License

MIT License — free to use, modify, and share.

---

## 👤 Author

[Your Name Here]
