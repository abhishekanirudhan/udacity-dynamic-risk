import os
import pandas as pd
import json
from sklearn.metrics import f1_score
import subprocess
import requests

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']

def main():
    
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt')) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}
        
    original_files = list(os.listdir(input_folder_path))
    
    if len(original_files) - len(ingested_files) == 0:
        return None
    
    ingestion.merge_multiple_dataframe()
    
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        deployed_score = float(f.readline())
        
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    
    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = df['exited'].values.reshape(-1, 1).ravel()
    
    y_pred = diagnostics.model_predictions(X)
    new_score = f1_score(y, y_pred)
    
    if new_score <= deployed_score:
        return None
    
    training.train_model()
    
    scoring.score_model()
    
    deployment.deploy_model()
    
    reporting.score_model()
    
    subprocess.run(['python', 'apicalls.py'])

if __name__ == '__main__':
    main()
