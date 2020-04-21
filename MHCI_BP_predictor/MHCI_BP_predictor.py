#MHCI_BP_predictor.py

import os, sys, math, re
import numpy as np
import pandas as pd
from time import time
import epitopepredict as ep

from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import PredictionFunction as PF

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

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

def build_predictor(training_data, allele, encoder, hidden_node):

    data = training_data.loc[training_data['allele'] == allele]
    if len(data) < 100:
        return

    # #write training dataframe to csv file
    # aw = re.sub('[*:]','_',allele) 
    # data.to_csv(os.path.join('alletes',aw+'_data.csv'))
    
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=5000, early_stopping=True,
                        activation='relu', solver='adam', random_state=2)
    X = data.peptide.apply(lambda x: pd.Series(encoder(x)),1) 
    y = data.log50k

    ## ---- TEST ---- ##
    # print(X)
    # print (allele, len(X))
    
    reg.fit(X,y)       
    return reg

def get_allele_names(data):
    a = data.allele.value_counts()
    a =a[a>200]
    return list(a.index)

def build_prediction_model(training_data, hidden_node):
    al = training_data.allele.unique().tolist()
    print(al)
    for a in al:
        aw = re.sub('[*:]','_', a) 
        fname = os.path.join(model_path, aw+'.joblib')
        reg = build_predictor(training_data, a, blosum_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s is done" %a)

def basicMHCiCrossValid(X, y, hidden_node):
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    auc_list = []
    r_list = []
    kf = KFold(n_splits=5, shuffle=True)
    for k, (train, test) in enumerate(kf.split(X, y)):
        print("Hidden nodee:%d, fold %d starts" %(hidden_node, k))
        t0 = time()
        reg.fit(X[train], y[train])
        scores = reg.predict(X[test])
        auc = PF.auc_score(y[test], scores, cutoff=.426)
        r = PF.pearson_score(y[test], scores)
        auc_list.append(auc)
        r_list.append(r)
        t1 = time()
        print("fold %d done" %k)
    # print(auc_list)
    # print(r_list)
    avg_auc = np.mean(auc_list)
    avg_r = np.mean(r_list)

    return avg_auc, avg_r

def test_Basic9merCrossValid():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[dataset['length'] == 9]
    alleles = dataset.allele.unique().tolist()
    # print(dataset)
    HiddenRange = range(60,61)
    header = pd.DataFrame(np.array(HiddenRange).reshape(1, -1), index=["hidden node"])
    header.to_csv(os.path.join(current_path, "basicMHC_One_crossValidation_auc.csv"), mode='a', header=False)
    header.to_csv(os.path.join(current_path, "basicMHC_One_crossValidation_pcc.csv"), mode='a', header=False)
    for allele in alleles:
        t0 = time()
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        X = allele_dataset.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1).to_numpy()
        y = allele_dataset.log50k.to_numpy()
        auc_list = []
        pcc_list = []
        for i in HiddenRange:
            auc, r = basicMHCiCrossValid(X, y, i)
            auc_list.append(auc)
            pcc_list.append(r)
            # score = pd.DataFrame(np.array([auc, r]).reshape(1, -1), columns=["AUC", "PCC"], index=[str(i)])
            # score.to_csv(os.path.join(current_path, "basicPan_crossValidation.csv"), mode='a', header=False)
        auc_df = pd.DataFrame(np.array(auc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele])
        pcc_df = pd.DataFrame(np.array(pcc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele])
        print(auc_df)
        print(pcc_df)
        auc_df.to_csv(os.path.join(current_path, "basicMHC_One_crossValidation_auc.csv"), mode='a', header=False)
        pcc_df.to_csv(os.path.join(current_path, "basicMHC_One_crossValidation_pcc.csv"), mode='a', header=False)
        t1 = time()
        print("%s is done, run in Elapsed time %d(m)" %(allele, (t1-t0)/60))


# test_Basic9merCrossValid()

def BuildModel(hidden_node):
    allmer_mhci = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(allmer_mhci)
    allele_dataset = dataset.loc[dataset['length'] == 9]
    alleles = allele_dataset.allele.unique().tolist()
    for allele in alleles:
        data = allele_dataset.loc[allele_dataset['allele'] == allele]
        aw = re.sub('[*:]','_', allele) 
        fname = os.path.join(model_path, aw+'.joblib')
        reg = build_predictor(data, allele, blosum_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s is done" %allele)

BuildModel(14)