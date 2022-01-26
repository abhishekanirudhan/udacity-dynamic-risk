import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

import diagnostics


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)
    
test_data_path = os.path.join(config['test_data_path'])

output_model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    filename = 'testdata.csv'
    filepath = os.path.join(test_data_path, filename)
    
    pred = diagnostics.model_predictions(filename)
    
    df = pd.read_csv(filepath)
    y_test = df['exited'].values.reshape(-1, 1)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))


if __name__ == '__main__':
    score_model()