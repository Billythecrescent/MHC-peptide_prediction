#MHCi2_prediction.py

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

def blosum62_encode(seq):
    return PF.encode(PF.readBLOSUM(62), seq)

def evaluate_predictor(X, y, allele):

    #print (len(data))
    # print(list(data.peptide), allele)
    reg = PF.find_model(allele, 9)
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
    if seqLen < 9:
        #turn 8mer to 9mer
        for i in range(seqLen-3):
            seq_list.append(seq[:(i+3)]+'X'*(9-seqLen)+seq[(i+3):])

    elif seqLen >= 10:
        bulg = seqLen - 9
        #turen 10mer to 9mer
        for i in range(3, 9):
            seq_list.append(seq[:i]+seq[(i+bulg):])
  
    seq_df = pd.DataFrame(seq_list, columns=['peptide'])
    # print(seq_df)

    #encode
    X = seq_df.peptide.apply(lambda x: pd.Series(blosum62_encode(x)),1)

    #find model
    aw = re.sub('[*:]','_',allele) 
    reg = PF.find_model(aw, 9)
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

    # print(predict_non9mer("HLA-A*01:01", "AQFSPQ"))
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
    pcc = PF.pearson_score(y, data_scores)
    # print(auc)

    return auc, pcc

def get_evaluation_by_allele():
    
    path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['length'] != 9]
    # print(dataset)
    alleles = dataset.allele.unique().tolist()
    header = pd.DataFrame(np.array(["AUC", "PCC"]).reshape(1, -1), index=["allele"])
    header.to_csv(os.path.join(current_path, "MHCi2_L-appro_scores.csv"), mode='a', header=False)
    print(alleles)
    for allele in alleles:
        t0 = time()
        data = dataset.loc[dataset['allele'] == allele]
        auc, pcc = LengthFree_predictor(allele, data)
        performance = pd.DataFrame(np.array([auc, pcc]).reshape(1, -1), columns=['AUC', 'PCC'], index=[allele])
        performance.to_csv(os.path.join(current_path, "MHCi2_L-appro_scores.csv"), mode='a', header=False)
        print(performance)
        t1 = time()
        print("%s is done, run in Elapsed time %d(m)" %(allele, (t1-t0)/60))
        
# get_evaluation_by_allele()

def mhci2_predictPeptide(dataset, outputFile=None):
    alleles = dataset.allele.unique().tolist()
    df_list = []
    # print(dataset)
    for allele in alleles:
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        data_9mer = allele_dataset.loc[allele_dataset['length'] == 9]
        data_non9mer = allele_dataset.loc[allele_dataset['length'] != 9]
        aw = re.sub('[*:]','_',allele) 
        reg = PF.find_model(aw, 9)
        X = data_9mer.peptide.apply(lambda x: pd.Series(blosum62_encode(x)),1)
        data_9mer_scores = pd.DataFrame(reg.predict(X), columns=['MHCi2_log50k'], index=data_9mer.index)
        data_non9mer_scores = data_non9mer.peptide.apply(lambda x: predict_non9mer(allele, x))
        data_non9mer_scores = data_non9mer_scores.to_frame()
        data_non9mer_scores.columns = ['MHCi2_log50k']
        
        result_9mer = pd.concat([data_9mer, data_9mer_scores], axis=1)
        result_non9mer = pd.concat([data_non9mer, data_non9mer_scores], axis=1)
        # print(result_9mer)
        # print(result_non9mer)
        df_list.append(result_9mer)
        df_list.append(result_non9mer)    

    combined_df = pd.concat(df_list, axis=0, sort=True)
    combined_df.sort_index(inplace=True)
    print(combined_df)

    if outputFile != None:
        combined_df.to_csv(outputFile)

def test_mhci2_predictPeptide():
    path = os.path.join(data_path, "modified_mhciTumor_dataset.csv")
    dataset = pd.read_csv(path)
    mhci2_predictPeptide(dataset, os.path.join(current_path, "mhci2_Tumor_result.csv"))

# test_mhci2_predictPeptide()