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

def blosum62_encode(seq):
    return PF.encode(PF.readBLOSUM(62), seq)

def build_predictor(training_data, allele, encoder, hidden_node):
    '''use taining data to train MLPRegressor model
    training_data: DataFrame
        the training data (must contain peptide and log50k column)
    allele: string
        the name of the allele, must be standard name
    encoder: function
        the encoding method used in the training
    hidden_node: int
        hidden node number
    
    Return:
    -------
    reg: MLPRegressor
        The fitted regressor model
    '''
    data = training_data.loc[training_data['allele'] == allele]

    #set data numebr threshold
    if len(data) < 100:
        return
    
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=5000, early_stopping=True,
                        activation='relu', solver='adam', random_state=2)
    X = data.peptide.apply(lambda x: pd.Series(encoder(x)),1) 
    y = data.log50k

    reg.fit(X,y)       
    return reg

def get_allele_names(data, threshold=100):
    '''find alleles in the data which have more data number than the threshold
    data: DataFrame
        the object data
    threshold: int
        data number (sample number) threshold
    
    Return:
    -------
    names: list
        the list of the alleles having more data than threshold
    '''
    a = data.allele.value_counts()
    a =a[a>threshold]
    names = list(a.index)
    return names

def basicMHCi_save_model(training_data, hidden_node):
    '''conduct basic (9mer) MHCi predictor building and save it to model_path.
    training_data: DataFrame
        the training data (must contain peptide and log50k column)
    hidden_node: int
        hidden node number

    Return:
    -------
    None
    '''
    alleles = training_data.allele.unique().tolist()
    # print(alleles)
    for allele in alleles:
        aw = re.sub('[*:]','_', allele) 
        fname = os.path.join(model_path, aw+'.joblib')
        reg = build_predictor(training_data, allele, blosum62_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s is done" %allele)

def basicMHCiCrossValid(X, y, hidden_node, cv_num):
    '''K-fold Cross Validation for basic MHCi1 method
    X: DataFrame
        encoded training input nodes (features)
    y: DataFrame
        labels
    hidden_node: int
        hidden node number
    cv_num: int
        the K number in K-fold CrossValidation (CV)

    Return:
    ------ 
    avg_auc: double
        average Area Under Curve (AUC) valie in K-fold CV
    avg_r: double
        average Pearson Correlation Coeefficient (PCC) value in k-fold CV
    '''
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    auc_list = []
    r_list = []
    kf = KFold(n_splits=cv_num, shuffle=True)
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
    '''Test and Record Basic9mer Cross Validation
    '''
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
        X = allele_dataset.peptide.apply(lambda x: pd.Series(blosum62_encode(x)),1).to_numpy()
        y = allele_dataset.log50k.to_numpy()
        auc_list = []
        pcc_list = []
        for i in HiddenRange:
            auc, r = basicMHCiCrossValid(X, y, i, 5)
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
        reg = build_predictor(data, allele, blosum62_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s is done" %allele)

# BuildModel(14)