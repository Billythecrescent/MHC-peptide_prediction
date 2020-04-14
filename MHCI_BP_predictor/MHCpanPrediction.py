'''
File: MHCpanPrediction.py
Author: Mucun Hou
Date: Apr 14, 2020
Description: This script is to predict MHC-I binding peptide based on a pan-spesific
    method, from NetMHCpan, but use different pseudo sequence, different dataset and 
    aim at different alleles.
'''

import os, re
import pandas as pd
import numpy as np
from time import time
# from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import *
import joblib
import epitopepredict as ep

import PredictionFunction as PF

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

blosum_encode = PF.blosum_encode

#pseudo sequence length = 40 
pseudo_sequence = [5, 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, \
 95, 97, 99, 114, 116, 118, 123, 124, 143, 146, 147, 150, 152, 155, 156, 158, 159, 163, 167, 170, 171]

# print(len(pseudo_sequence))

