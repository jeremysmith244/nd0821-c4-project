import pathlib
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'], config['model_fname']) 
test_data_path = os.path.join(config['test_data_path'], config["testdata_fname"])
log_file_path = os.path.join(config['output_model_path'], config['log_fname']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values
    y_test = test_data['exited'].values
    log = joblib.load(model_path)
    y_pred = log.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    with open(log_file_path, 'w') as f:
        f.write(str(f1))

if __name__ == '__main__':
    score_model()