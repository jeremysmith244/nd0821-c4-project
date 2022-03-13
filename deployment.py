from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

file_list_path = os.path.join(config['output_folder_path'], config['datalist_fname'])
model_path = os.path.join(config['output_model_path'], config['model_fname']) 
log_file_path = os.path.join(config['output_model_path'], config['log_fname']) 
prod_filelist_path = os.path.join(config['prod_deployment_path'], config['datalist_fname']) 
prod_model_path = os.path.join(config['prod_deployment_path'], config['model_fname']) 
prod_logfile_path = os.path.join(config['prod_deployment_path'], config['log_fname']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(file_list_path, prod_filelist_path)
    shutil.copy(model_path, prod_model_path)
    shutil.copy(log_file_path, prod_logfile_path)    
        

if __name__ == '__main__':
    store_model_into_pickle()        

