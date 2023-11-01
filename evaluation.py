#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

def evaluate(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))

if __name__=='__main__':
    model_path = os.path.join('.', 'xgboost-model')
    test_path = 'processed_data.csv'  # path to the test data
    output_evaluation_path = os.path.join('.', 'evaluation.json')

    booster = xgb.Booster()
    booster.load_model(model_path)

    test_data = pd.read_csv(test_path)
    y_test = test_data.iloc[:, 0].to_numpy()  # assuming the first column is the target variable
    X_test = test_data.iloc[:, 1:].to_numpy()

    dtest = xgb.DMatrix(X_test)
    predictions = booster.predict(dtest)

    rmse = evaluate(predictions, y_test)

    with open(output_evaluation_path, 'w') as file:
        file.write(json.dumps({'RMSE': rmse}))

