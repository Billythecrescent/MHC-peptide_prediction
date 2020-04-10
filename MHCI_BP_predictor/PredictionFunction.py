'''
File: PredictionFunction.py
Author: Mucun Hou
Date: Apr 10, 2020
Description: This script integrates and provides prediction-used function
    for other use.
'''

import os.path, re
from math import log
import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
import epitopepredict as ep

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def dataset2fasta(dataset, filePath):
    '''Convert MHC affinity DataFrame to fasta file
    dataset: DataFrame
        MHC affinity data, must contain 'allele' and 'peptide' columns
    filename: string
        The output file path
    
    Return:
    ------
    True
    '''
    f = open(filePath,'w')
    # peptides = dataset.peptide
    # alleles = dataset.allele
    # datapair = dict(zip(alleles,peptides)) 
    # print(datapair)
    for index, sample in dataset.iterrows():
        # print(sample['allele'])
        # print(sample['peptide'])
        print('>' + sample['allele'], file = f)
        print(sample['peptide'] + '\n', file = f)
    f.close()

    return True

def blosum_encode(seq):
    '''
    Encode protein sequence, seq, to one-dimension array.
    Use blosum62 matrix to encode the number.
    input: [string] seq (length = n)
    output: [1x24n ndarray] e
    '''
    #encode a peptide into blosum features
    s=list(seq)
    blosum62 = ep.blosum62
    x = pd.DataFrame([blosum62[i] for i in seq]).reset_index(drop=True)
    e = x.to_numpy().flatten() 
    # print(x)   
    return e

def geo_mean(iterable):
    nplist = np.array(iterable)
    return nplist.prod()**(1.0/len(nplist))

def auc_score(true,sc,cutoff = None):
    '''
    Calculate the auc score of soc curve
    '''
    if cutoff!=None:
        true = (true<=cutoff).astype(int)
        sc = (sc<=cutoff).astype(int)
    else:
        print("Please specify the classcification threshould!")
        return 
    # print(true, sc)
    
    if len(np.unique(true)) == 1: # bug in roc_auc_score
        r =  metrics.accuracy_score(true, np.rint(sc))
        return r
    r = metrics.roc_auc_score(true, sc) 
    # #Or use the following code for alternative
    # fpr, tpr, thresholds = metrics.roc_curve(true, sc, pos_label=1)
    # r = metrics.auc(fpr, tpr)
    
    return  r

def find_model(allele, length):
    '''Find model for alleles of different lengths. 9mer: ../model/  non9mer: ../model/Non9mer/
    SHOULD "import joblib" first
    allele: string
        standardized allele name (by regex according to the prediction method)
        It is different from true allele because Windows os file system
    length: int
        the length of the inquery peptide

    Return
    ------
    reg: MLPRegressor
        the regression predictor
    '''
    if length != 9:
        fname = os.path.join(os.path.join(model_path, "Non9mer"), allele + "-" + str(length) +'.joblib')
    elif length == 9:
        fname = os.path.join(model_path, allele+'.joblib')
    print(fname)
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return

