#MHCI_BP_evaluator.py

import os, sys, math, re
import numpy as np
import pandas as pd
from time import time
import epitopepredict as ep

from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor

import PredictionFunction as PF

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

blosum_encode = PF.blosum_encode

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
    auc = PF.auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def predict_non9mer(allele, seq):

    seq_list = []
    seqLen = len(seq)
    if seqLen == 8:
        #turn 8mer to 9mer
        for i in range(5):
            seq_list.append(seq[:(i+3)]+'X'+seq[(i+3):])

    elif seqLen >= 10:
        bulg = seqLen - 9
        #turen 10mer to 9mer
        for i in range(3, 9):
            seq_list.append(seq[:i]+seq[(i+bulg):])
  
    seq_df = pd.DataFrame(seq_list, columns=['peptide'])
    # print(seq_df)

    #encode
    X = seq_df.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)

    #find model
    aw = re.sub('[*:]','_',allele) 
    reg = find_model(aw)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return 0
    
    #predict
    scores = reg.predict(X)
    # print(scores)

    #geometric mean
    mean_score = PF.geo_mean(scores)
    # print(mean_score, seq)
    return mean_score


def test_predict_non9mer():
    path = os.path.join(data_path, "modified_mhc.20130222.csv")
    allele = 'HLA-A*01:01'
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['length'] == 8]
    dataset = dataset.loc[dataset['allele'] == allele]
    # print(dataset)
    dataset.peptide.apply(lambda x: predict_non9mer(allele, x))

    # print(predict_non9mer("HLA-A*01:01", "AQFSPQYL"))
    # print(predict_non9mer("HLA-A*01:01", "YSLEYFQFVKK"))
    # print(predict_non9mer('HLA-A*01:01', "ABCDEABCDEABCDEABCDEABCDEABCDE"))

# test_predict_non9mer()

def LengthFree_predictor(allele, data):

    y = data.log50k
    
    data_scores = data.peptide.apply(lambda x: predict_non9mer(allele, x))
    data_scores.columns = ['score']
    # print(data_scores)

    #Generate auc value
    auc = PF.auc_score(y, data_scores,cutoff=.426)
    # print(auc)

    return auc

## Primary Test ##
# allele = "HLA-A*01:01"
# test_data_8 = ep.get_training_set(allele, length=8)
# test_data_10 = ep.get_training_set(allele, length=10)
# test_data_11 = ep.get_training_set(allele, length=11)
# LengthFree_predictor(allele, test_data_11)

## Secondary Test ##
# dataset_filename = os.path.join(data_path, "evalset_8mer_normalization.csv")
# df = pd.read_csv(dataset_filename)
# alleles = df.allele.unique()
# # print(alleles)
# allele = alleles[12]
# print(allele)
# data = df.loc[df['allele'] == allele]
# result = LengthFree_predictor(allele, data)


def get_evaluation_by_allele():
    
    path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['length'] != 9]
    # print(dataset)
    alleles = dataset.allele.unique().tolist()
    header = pd.DataFrame(np.array(["AUC"]).reshape(1, -1), index=["allele"])
    header.to_csv(os.path.join(current_path, "MHCi2_L-appro_scores.csv"), mode='a', header=False)
    print(alleles)
    for allele in alleles:
        t0 = time()
        data = dataset.loc[dataset['allele'] == allele]
        auc = LengthFree_predictor(allele, data)
        performance = pd.DataFrame(np.array([auc]).reshape(1, -1), columns=['AUC'], index=[allele])
        performance.to_csv(os.path.join(current_path, "MHCi2_L-appro_scores.csv"), mode='a', header=False)
        print(performance)
        t1 = time()
        print("%s is done, run in Elapsed time %d(m)" %(allele, (t1-t0)/60))
        
# get_evaluation_by_allele()
    