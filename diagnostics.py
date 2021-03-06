
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_folder_path = os.path.join(config['output_folder_path'])

##################Function to get model predictions
def model_predictions(X_df):
    #read the deployed model and a test dataset, calculate predictions
    file = open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb')
    trained_model = pickle.load(file)
    file.close()
    
    pred = trained_model.predict(X_df)
    
    return pred

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'))
    df = df.drop(['exited'], axis = 1)
    df = df.select_dtypes('number')
    
    summary_stats = {}
    
    for col in df.columns:
        mean = df[col].mean()
        median = df[col].median()
        sd = df[col].std()
        
        summary_stats[col] = [mean, median, sd]
    
    return summary_stats

def missing_data():
    df = pd.read_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'))
    
    missing_list = list(df.isna().sum())
    missing_pct = [missing_list[i]/len(df.index) for i in range(len(missing_list))]
    
    return missing_pct

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    subprocess.run(['python', 'ingestion.py'])
    ingest_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    subprocess.run(['python', 'training.py'])
    train_time = timeit.default_timer() - start_time

    time_list = [ingest_time, train_time]
    
    return time_list

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.run(['python', '-m', 'pip', 'list', '--outdated']).stdout
    return outdated


if __name__ == '__main__':
    model_predictions('testdata.csv')
    dataframe_summary()
    execution_time()
    outdated_packages_list()