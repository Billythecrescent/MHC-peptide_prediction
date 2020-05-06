#MHCi1_prediction.py

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

def basicMHCi_save_model(training_data, length, hidden_node):
    '''conduct basic (9mer) MHCi predictor building and save it to model_path.
    training_data: DataFrame
        the training data (must contain peptide and log50k column)
    length: int
        the length of the peptides in training_data
    hidden_node: int
        hidden node number

    Return:
    -------
    None
    '''
    alleles = training_data.allele.unique().tolist()
    path = os.path.join(model_path, "mhci1")
    for allele in alleles:
        aw = re.sub('[*:]','_', allele) 
        fname = os.path.join(path, aw + "-" + str(length) +'.joblib')
        reg = build_predictor(training_data, allele, blosum62_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s on length %d is done" %(allele, length))

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

def MHCi1_allLength_CrossValid(file_path, length):
    '''Test and Record Basic9mer Cross Validation
    '''
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[dataset['length'] == length]
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

# MHCi1_allLength_CrossValid(os.path.join(data_path, "modified_mhc.20130222.csv"), 9)

def mhci1_predictPeptide(dataset, outputFile=None):
    dataset = pd.DataFrame(dataset.query('length > 7 and length < 12'))
    alleles = dataset.allele.unique().tolist()
    df_list = []
    print(dataset)
    for allele in alleles:
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        lengths = allele_dataset.length.unique().tolist()
        for length in lengths:
            data = allele_dataset.loc[allele_dataset['length'] == length]
            if len(data) == 0:
                continue
            aw = re.sub('[*:]','_',allele) 
            reg = PF.find_model(aw, length)
            if reg == None:
                scores = pd.DataFrame(np.array(['nan']*len(data)), columns=['MHCi1_log50k'], index=data.index)
            else:
                X = data.peptide.apply(lambda x: pd.Series(blosum62_encode(x)),1)
                # print(data.shape, X.shape)
                scores = pd.DataFrame(reg.predict(X), columns=['MHCi1_log50k'], index=data.index)
            result = pd.concat([data, scores], axis=1)
            df_list.append(result)
    combined_df = pd.concat(df_list, axis=0, sort=True)
    combined_df.sort_index(inplace=True)
    print(combined_df)

    if outputFile != None:
        combined_df.to_csv(outputFile)

def test_mhci1_predictPeptide():
    # path = os.path.join(data_path, "VACV_evaluation_dataset.csv")
    path = os.path.join(data_path, "modified_mhciTumor_dataset.csv")
    dataset = pd.read_csv(path)
    # mhci1_predictPeptide(dataset, os.path.join(current_path, "mhci1_VACV_result.csv"))
    mhci1_predictPeptide(dataset, os.path.join(current_path, "mhci1_Tumor_result.csv"))

# test_mhci1_predictPeptide()

def MHCi1_BuildModel_For_SingleLength(dataset, length, hidden_node):
    '''Find 9mer data in the dataset and use it to train and save predictor model
    Object path: model_path
    dataset: DataFrame
        The standardlized-format dataset, must contains 'allele', 'peptide', 'length', 'log50k' columns
    length: int
        The length of the peptides in training_data
    hidden_node: int
        The number of hidden layer nodes
    
    Return:
    ------
    None
    '''
    allele_dataset = dataset.loc[dataset['length'] == length]
    alleles = allele_dataset.allele.unique().tolist()
    for allele in alleles:
        data = allele_dataset.loc[allele_dataset['allele'] == allele]
        aw = re.sub('[*:]','_', allele) 
        fname = os.path.join(model_path, aw+'.joblib')
        reg = build_predictor(data, allele, blosum62_encode, hidden_node)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("predictor for allele %s is done" %allele)

def SingleEvaluation(X, y, allele, length):
    '''Evaluation of predictor of single allele on single length
    X: DataFrame
        encoded training input nodes (features)
    y: DataFrame
        labels
    length: int
        The length of the peptides being tested.

    '''
    reg = PF.find_model(allele, length)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return float('nan')
    scores = reg.predict(X)

    #Generate auc value
    auc = PF.auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def AlleleSpecificEvaluation(datasetPath, length):
    '''Evaluation of predictors of single allele on all lengths in the dataset
    datasetPath: string
        The file path of the dataset
    length: int
        The length of the peptides in the dataset
    
    Returns:
    -------
    auc_df: DataFrame
        The Area Under Curve (AUC) dataframe as output
    '''
    auc_list = []
    df = pd.read_csv(datasetPath)
    alleles = df.allele.unique().tolist()
    lengths = df.length.unique().tolist()
    if len(lengths) != 1:
        print("The dataset does not contain only one length")
        return
    if lengths[0] != length:
        print("The length of peptides in the dataset is not consistent with what you claimed.")
        return
    
    print("Your dataset has %d alleles.\n They are:" %len(alleles))
    print(alleles)

    for allele in alleles:
        data = df.loc[df['allele'] == allele]
        X = data.peptide.apply(lambda x: pd.Series(PF.blosum_encode(x)),1)
        y = data.log50k
        aw = re.sub('[*:]','_',allele) 
        result = SingleEvaluation(X, y, aw, length)
        auc_list.append(result)
        print("Evaluation: allele %s of length %d is done\n" %(allele, length))

    #Write Result
    auc_df = pd.DataFrame(auc_list, index = alleles.tolist())
    auc_df.columns = ['auc']
    print(auc_df)

    return auc_df

def Non9merCrossValid(X, y, hidden_node):
    
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    avg_auc_list = []
    avg_pcc_list = []
    for i in range(10):
        auc_list = []
        r_list = []
        kf = KFold(n_splits=5, shuffle=True)
        for k, (train, test) in enumerate(kf.split(X, y)):
            t0 = time()
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            r = PF.pearson_score(y[test], scores)
            auc_list.append(auc)
            r_list.append(r)
            t1 = time()
        # print(auc_list)
        # print(r_list)
        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        avg_auc_list.append(avg_auc)
        avg_pcc_list.append(avg_r)

    return np.mean(avg_auc_list), np.mean(avg_pcc_list)

def test_Non9merCrossValid():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[dataset['length'] != 9]
    alleles = dataset.allele.unique().tolist()
    # print(dataset)
    HiddenRange = range(5,21)
    header = pd.DataFrame(np.array(HiddenRange).reshape(1, -1), index=["hidden node"])
    header.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_auc.csv"), mode='a', header=False)
    header.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_pcc.csv"), mode='a', header=False)
    for length in [8, 10, 11]:
        length_dataset = dataset.loc[dataset['length'] == length]
        t0 = time()
        for allele in alleles:
            allele_dataset = length_dataset.loc[length_dataset['allele'] == allele]
            X = allele_dataset.peptide.apply(lambda x: pd.Series(blosum62_encode(x)),1).to_numpy()
            y = allele_dataset.log50k.to_numpy()
            auc_list = []
            pcc_list = []
            for i in HiddenRange:
                auc, r = Non9merCrossValid(X, y, i)
                auc_list.append(auc)
                pcc_list.append(r)
                # score = pd.DataFrame(np.array([auc, r]).reshape(1, -1), columns=["AUC", "PCC"], index=[str(i)])
                # score.to_csv(os.path.join(current_path, "basicPan_crossValidation.csv"), mode='a', header=False)
            auc_df = pd.DataFrame(np.array(auc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele+"_"+str(length)+"mer"])
            pcc_df = pd.DataFrame(np.array(pcc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele+"_"+str(length)+"mer"])
            print(auc_df)
            print(pcc_df)
            auc_df.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_auc.csv"), mode='a', header=False)
            pcc_df.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_pcc.csv"), mode='a', header=False)
            print("allele %s on length %d is done" %(allele, length))
        t1 = time()
        print("length %d is done, run in Elapsed time %d(m)" %(length, (t1-t0)/60))


# test_Non9merCrossValid()

def MHCi1_Build_Model_For_Lengths(DataPath, lengths, hidden_node):
    '''Build prediction models for data of different lengths

    '''
    dataset = pd.read_csv(DataPath)
    for length in lengths:
        length_dataset = dataset.loc[dataset['length'] == length]
        if len(length_dataset) > 20:
            MHCi1_BuildModel_For_SingleLength(length_dataset, length, hidden_node)

# MHCi1_Build_Model_For_Lengths(os.path.join(data_path, "modified_mhc.20130222.csv"), [8, 10, 11, 12, 13, 14], 14)
