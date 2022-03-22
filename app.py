from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from diagnostics import model_predictions
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, outdated_packages_list
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data_path = request.args.get('data_path')
    data_file = pd.read_csv(data_path)
    return str(model_predictions(data_file))

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    data_path = request.args.get('data_path')
    data_file = pd.read_csv(data_path)
    f1 = score_model(data_file)
    return str(f1)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    sum_stat = dataframe_summary(request.args.get('data_path'))
    return str(sum_stat)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    times = execution_time()
    versions = outdated_packages_list()
    return str(times) + '\n' + str(versions)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
