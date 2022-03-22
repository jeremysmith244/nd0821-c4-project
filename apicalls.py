import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'], config["testdata_fname"])
api_path = os.path.join(config['output_model_path'], config["testdata_fname"])


#Call each API endpoint and store the responses
response1 = requests.get(URL + '/prediction?data_path=%s'%test_data_path).content
response2 = requests.get(URL + '/scoring?data_path=%s'%test_data_path).content
response3 = requests.get(URL + '/summarystats?data_path=%s'%test_data_path).content
response4 = requests.get(URL + '/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open(config["api_save"], "w") as f:
    for response in responses:
        f.write(response.decode('ascii') + "\n")


