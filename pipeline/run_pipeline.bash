#!/bin/bash

echo "Starting full pipeline"

echo "running load_clean.py"
python load_clean.py

sleep 1
echo "running summarize_data.py"
python summarize_data.py

sleep 1
echo "running preprocess_model_data.py"
python preprocess_model_data.py

sleep 1
echo "running train_models.py"
python train_models.py

sleep 1
echo "running test_models.py"
python test_models.py

sleep 1
echo "running evaluate_models.py"
python evaluate_models.py 
