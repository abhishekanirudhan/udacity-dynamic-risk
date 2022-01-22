import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df = pd.DataFrame(columns = ['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])
    filenames = []
    
    for file in os.listdir(os.path.join(os.getcwd(), input_folder_path)):
        filenames.append(file)
        
        df_part = pd.read_csv(os.path.join(os.getcwd(), input_folder_path, file))
        df = df.append(df_part)
    
    # Dedupe data
    dedupe_df = df.drop_duplicates()
    
    dedupe_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index = False)
    
    record_file = open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w')
    for file in filenames:
        record_file.write(str(file) + '\n')
        

if __name__ == '__main__':
    merge_multiple_dataframe()
