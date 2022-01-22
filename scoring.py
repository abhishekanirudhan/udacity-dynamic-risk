from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    """
    Function accepts a trained model, loads test data, and calculates 
    an F1 score for the model relative to the test data.
    It then writes the result to the latestscore.txt file
    """
    
    # Load in test data
    testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    
    # Reshape data
    X_test = testdata.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y_test = testdata['exited'].values.reshape(-1, 1)
    
    # Load the model
    with open(os.path.join(os.getcwd(), output_model_path, 'trainedmodel.pkl'), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    
    # Get predictions
    pred = model.predict(X_test)
    
    # Get F1 score
    f1score = metrics.f1_score(pred, y_test)
    print(f1score)
    
    with open(os.path.join(os.getcwd(), output_model_path, 'latestscore.txt'), 'w') as latestscore:
        latestscore.write(str(f1score) + '\n')

if __name__ == '__main__':
    score_model()