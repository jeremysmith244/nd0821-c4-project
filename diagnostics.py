
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
from urllib.request import urlopen
import json

def getLatestVersion(pkgName):
    
    contents = urlopen('https://pypi.org/pypi/'+pkgName+'/json').read()
    data = json.loads(contents)
    latest_version = data['info']['version']

    return latest_version

def ingestion_timing():
    start = timeit.default_timer()
    os.system("python ingestion.py")
    stop = timeit.default_timer()
    return stop-start

def training_timing():
    start = timeit.default_timer()
    os.system("python training.py")
    stop = timeit.default_timer()
    return stop-start

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], config["traindata_fname"]) 
test_data_path = os.path.join(config['test_data_path'])
prod_model_path = os.path.join(config['prod_deployment_path'], config['model_fname']) 

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    log = joblib.load(prod_model_path)
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values
    return log.predict(X)

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data = pd.read_csv(dataset_csv_path)
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except:
            data = data.drop(columns=[col])
    means = data.mean()
    medians = data.median()
    stds = data.std()
    na_percent = (data.isna().sum() / data.shape[0]).to_list()
    return [means, medians, stds, na_percent]

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    ingest_time = ingestion_timing()
    train_time = training_timing()
    return [ingest_time, train_time]

##################Function to check dependencies
def outdated_packages_list():
    pack_summary = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            cur_pck, cur_v = line.split("==")
            lts_v = getLatestVersion(cur_pck)
            pack_summary.append([cur_pck.strip(), cur_v.strip(), lts_v])
    pack_summary = pd.DataFrame(pack_summary, columns=['package', 'current_version', 'latest_version'])
    return pack_summary


if __name__ == '__main__':
    data = pd.read_csv(dataset_csv_path)
    model_predictions(data)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
