from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess

import diagnostics


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    #filepath = request.get_json()[]
    pass #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scores():        
    #check the score of the deployed model
    scores = subprocess.run(['python', 'scoring.py'], capture_output = True).stdout
    return scores

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    #return a list of all calculated summary statistics
    summary = diagnostics.dataframe_summary()
    
    return jsonify(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    #add return value for all diagnostics
    missing = diagnostics.missing_data()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()
    
    output = {'missing data pct': missing,
             'execution time': time,
             'outdated packages': outdated}
    
    return jsonify(output)

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)