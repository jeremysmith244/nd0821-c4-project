from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], config["traindata_fname"]) 
model_path = os.path.join(config['output_model_path'], config['model_fname']) 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    log = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, l1_ratio=None, max_iter=100,
                             multi_class='multinomial', n_jobs=None, penalty='l2',
                             random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                             warm_start=False)
    
    #fit the logistic regression to your data
    data = pd.read_csv(dataset_csv_path)
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values
    y = data['exited'].values

    log.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    joblib.dump(log, model_path)

if __name__ == "__main__":
    train_model()
