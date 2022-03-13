import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], config["testdata_fname"])
model_path = os.path.join(config['prod_deployment_path'], config['model_fname'])
cm_path = os.path.join(config['output_model_path'], config['cm_fname'])


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data = pd.read_csv(test_data_path)
    y_pred = model_predictions(test_data)
    y = test_data['exited'].values
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8,8))
    plt.imshow(cm)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(cm_path)
    plt.close();


if __name__ == '__main__':
    score_model()
