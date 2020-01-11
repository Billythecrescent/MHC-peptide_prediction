#MHCI_BP_evaluator.py

import os, sys, math, re
import numpy as np
import pandas as pd
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
import epitopepredict as ep

from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor

module_path = os.path.dirname(os.path.abspath(__file__)) #path to module
model_path = os.path.join(os.path.abspath(os.path.dirname(module_path)),"models")

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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

def auc_score(true,sc,cutoff=None):
    '''
    Calculate the auc score of soc curve
    '''
    if cutoff!=None:
        true = (true<=cutoff).astype(int)
        sc = (sc<=cutoff).astype(int)
    # print(true, sc)
    
    r = metrics.roc_auc_score(true, sc) 
    # #Or use the following code for alternative
    # fpr, tpr, thresholds = metrics.roc_curve(true, sc, pos_label=1)
    # r = metrics.auc(fpr, tpr)
    
    return  r

def find_model(allele):
    fname = os.path.join(model_path, allele+'.joblib')
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return

def evaluate_predictor(X, y, allele):

    #print (len(data))
    # print(list(data.peptide), allele)
    reg = find_model(allele)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return
    scores = reg.predict(X)

    #Generate auc value
    auc = auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def main():
    
    alleles = ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:06"]
    comp=[]
    evalset = ep.get_evaluation_set(length=9) #type: DataFrame

    # #write training dataframe to csv file
    # evalset.to_csv(os.path.join('evaluate_data.csv'))
    
    test_alleles = evalset.allele.unique() #numpy.ndarray 'str'
    # print(test_alleles)
    for allele in test_alleles:
        data = ep.get_evaluation_set(allele, length=9)
        X = data.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)
        y = data.log50k
        aw = re.sub('[*:]','_',allele) 
        result = evaluate_predictor(X, y, aw)
        # print(result)
        comp.append(result)
    fo = open("written_result.csv",'w')
    print(test_alleles, file=fo)
    print(comp, file=fo)



main()