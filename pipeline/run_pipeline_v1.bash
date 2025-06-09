#!/bin/bash

echo "Starting full pipeline"

echo "running 01_load_clean.py"
python 01_load_clean.py

sleep 1
echo "running 02_summarize_data.py"
python 02_summarize_data.py

sleep 1
echo "running 03_preprocess_model_data.py"
python 03_preprocess_model_data.py

sleep 1
echo "running 04_train_models_<method>.py"
python 04_train_models_optuna.py
#python 04_train_models_GridSearchCV.py
#python 04_train_models_RandomizedSearchCV.py

sleep 1
echo "running 05_vis_hp_results.py"
python 05_vis_hp_results.py

sleep 1
echo "running 06_test_models.py"
python 06_test_models.py

sleep 1
echo "running 07_evaluate_models.py"
python 07_evaluate_models.py 

sleep 1
echo "running 08_feature_importance.py"
python 08_feature_importance.py

