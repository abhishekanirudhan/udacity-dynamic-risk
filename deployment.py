
import pickle
import os
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def deploy_model():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    # Copy ingestedfiles.txt
    shutil.copy(os.path.join(os.getcwd(), dataset_csv_path, 'ingestedfiles.txt'),
               os.path.join(os.getcwd(), prod_deployment_path))
    
    # Copy latestscore.txt
    shutil.copy(os.path.join(os.getcwd(), output_model_path, 'latestscore.txt'),
               os.path.join(os.getcwd(), prod_deployment_path))
    
    # Copy trainedmodel.pkl
    shutil.copy(os.path.join(os.getcwd(), output_model_path, 'trainedmodel.pkl'),
               os.path.join(os.getcwd(), prod_deployment_path))

if __name__ == '__main__':
    deploy_model()