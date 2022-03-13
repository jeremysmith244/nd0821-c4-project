import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path'], config["traindata_fname"])
file_list_path = os.path.join(config['output_folder_path'], config['datalist_fname']) 


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    inpt_folder = Path(input_folder_path)
    files = list(inpt_folder.glob('*.csv'))
    summary = pd.DataFrame([])
    ingested = []
    for file in files:
        ingested.append(file.name)
        if summary.empty:
            summary = pd.read_csv(file)
        else:
            summary = summary.append(pd.read_csv(file), ignore_index=True)
    summary = summary.drop_duplicates()
    summary.to_csv(dataset_csv_path)
    with open(file_list_path, 'w') as f:
        for line in ingested:
            f.write(line + '\n')

if __name__ == '__main__':
    merge_multiple_dataframe()
