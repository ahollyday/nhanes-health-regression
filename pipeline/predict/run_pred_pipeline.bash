#!/bin/bash

echo "starting prediction pipeline"

echo "running 01_load_clean_pred.py"
python 01_load_clean_pred.py
sleep 1

echo "running 02_summarize_data_pred.py"
python 02_summarize_data_pred.py
sleep 1

echo "running 03_preprocess_pred.py"
python 03_preprocess_pred.py
sleep 1

echo "running 04_predict.py"
python 04_predict.py
sleep 1

