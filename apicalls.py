import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

#Call each API endpoint and store the responses
response_pred = requests.post(f'{URL}/prediction', 
                             json = {
                             'path': test_data_path}).text

response_score = requests.get(f'{URL}/scoring').text
response_stats = requests.get(f'{URL}/summarystats').text
response_diagnose = requests.get(f'{URL}/diagnostics').text


#write the responses to your workspace
with open(os.path.join(output_model_path, 'apiresponses.txt'), 'w') as file:
    file.write('Model Predictions\n')
    file.write(response_pred)
    file.write('\n\nF1 Score\n')
    file.write(response_score)
    file.write('\n\nSummary Statistics\n')
    file.write(response_stats)
    file.write('\n\Diagnostics\n')
    file.write(response_diagnose)


