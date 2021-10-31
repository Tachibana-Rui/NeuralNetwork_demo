import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics


# Classic Custom Datset class with 

# __init__ , __len__, and __getitem__
class SignalDataset(Dataset):
    
    def __init__(self, train=True):
        
        self.train_dataset = train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check whether GPU is avaliable, else it will use CPU

        if self.train_dataset: 
            self.file_fname_set = sorted(glob.glob('1d_signal_data/train/*.csv'))
        else: 
            self.file_fname_set = sorted(glob.glob('1d_signal_data/test_internal/*.csv'))
            
    
    def __len__(self):
        # calc length of total choicen dataset and return.
        return len(self.file_fname_set)
        
        
    def __getitem__(self, indx): 
        
        
        if self.train_dataset:
            # S: Load data item into a numpy array
            x = pd.read_csv(self.file_fname_set[indx], header=None).to_numpy().flatten() # Confusing for students - WARN and specifcy output.

            # S: Extract Class Label from name
            path_fname = self.file_fname_set[indx] #
            fname = path_fname.split('/')[-1] # split path by / and get the file name (last element)

            label = fname.split('_')[-1].split('.')[0] # split and pick operations to get the label from eg "ID_XX_0.csv"
            label = int(label)

            # Transfrom X into a proper tensor # MAJ NOTES: Mention that type is important for PytOrch-> Float
            x_tensor = torch.tensor(x).float().to(self.device) #Changed here

            # Return X with Y (trani) or return just X in case of test-set.
            return x_tensor, label
        
        else: 
            # S: Load data item into a numpy array
            x = pd.read_csv(self.file_fname_set[indx], header=None).to_numpy().flatten() # Confusing for students - WARN and specifcy output.

            # Transfrom X into a proper tensor # MAJ NOTES: Mention that type is important for PytOrch-> Float
            x_tensor = torch.tensor(x).float().to(self.device) #Changed here
            
            return x_tensor
    
