

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import sys
from pathlib import Path
import subprocess
import pandas as pd

with open('config.json','r') as f:
    config = json.load(f) 

##################Check and read new data
#first, read ingestedfiles.txt
file_list_path = os.path.join(config["prod_deployment_path"], config['datalist_fname'])
ingested_files = []
with open(file_list_path, 'r') as f:
    for line in f:
        ingested_files.append(line)
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
data_files = list(Path(config["input_folder_path"]).glob('*.csv'))

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

new_files = False
for file in data_files:
    for complete in ingested_files:
        if complete != file.name:
            new_files = True
            break
if not new_files:
    print("No new data found, exiting!")
    sys.exit()

print("Found new data!")
subprocess.run(args=['python', 'ingestion.py'])

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
old_file_path = os.path.join(config["prod_deployment_path"], config['log_fname']) 
with open(old_file_path, 'r') as f:
    score = float(f.readline())

new_data = pd.read_csv(data_files[0])
new_score = scoring.score_model(new_data, production=True)
new_score = 0.1

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
print('New Score: ' + str(new_score))
print('Old Score: ' + str(score))
if new_score < score:
    print('Model drift has occured, retraining...')
else:
    print('Model has not drifted, exiting...')
    sys.exit()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.run(args=['python', 'training.py'])
subprocess.run(args=['python', 'deployment.py'])

print('Model retrained and deployed!')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.run(args=['python', 'apicalls.py'])
subprocess.run(args=['python', 'reporting.py'])
