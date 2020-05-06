'''
File: MHCpanPrediction.py
Author: Mucun Hou
Date: Apr 18, 2020
Description: This script is to read and process MHC-II data from file.
'''

import os, re
import pandas as pd
import numpy as np
from math import log
import joblib
from sklearn import metrics
from scipy.stats import pearsonr
import epitopepredict as ep

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
mhcii_path = os.path.join(data_path, "mhcii")

def readBLOSUM(index = 0):
    '''read BLOSUM matrix to DataFrame by calling its index. eg: 50, 62. Type readBLOSUM() for help
    index: int
        The index of BLOSUM matrix, 62 for blosum62
    return DataFrame
        The BLOSUM matrix DataFrame

    '''
    matricesList = [i for i in [40, 50, 55, 60 ,62, 65, 70, 75, 80, 85, 90]]
    if index == 0:
        print("Read BLOSUM matrix as DataFrame.\nAvailable index:", matricesList)
        return 
    filepath = os.path.join(matrices_path, "BLOSUM"+str(index)+".txt")
    # header = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
    blosum = pd.read_csv(filepath, header=0, index_col=0)
    return blosum

def readData(dirname):
    '''read mhcii affinity data from dirname directory. 
    NOTE: no other files be permitted in this dir
    dirname: string
        the name of the directory to be searched
    
    Return:
    -------
    Dataset: pd.DataFrame
        the accumulated dataset of all files in thie dir.
    
    '''
    fileList = os.listdir(dirname)
    dataset_list = []
    for filename in fileList:
        dataset = pd.read_csv(os.path.join(dirname, filename), sep="\t", header=None)
        dataset.columns = ["species", "allele", "length", "peptide_source", "peptide", "inequalty", "ic50"]
        # dataset = pd.DataFrame(dataset, columns=["species", "allele", "length", "peptide_source", "inequalty", "ic50"])
        # print(dataset)
        dataset_list.append(dataset)
    
    Dataset = dataset_list[0]
    for i in range(1, len(dataset_list)):
        Dataset = pd.concat([Dataset, dataset_list[i]], 0)
    print(Dataset)

    return Dataset
    
# readData(mhcii_path)

def datasetOutput(dataset, format = None, output_filename = None):
    '''output the dataset to csv format
    '''
    if format == "csv":
        if output_filename == None:
            dataset.to_csv(os.path.join(module_path, "dataset_output.csv"))
        elif output_filename != None:
            dataset.to_csv(os.path.join(module_path, output_filename + ".csv"))

def test_datasetOutput():
    dataset = readData(mhcii_path)
    # datasetOutput(dataset, 'csv', os.path.join(data_path, "mhcii-DRB1-dataset"))
    datasetOutput(dataset, 'csv', os.path.join(data_path, "mhcii-XXX-dataset"))

# test_datasetOutput()

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

def encode(matix, seq):
    '''
    Encode protein sequence, seq, to one-dimension array.
    Use blosum matrix to encode the number.
    input: [string] seq (length = n)
    output: [1x24n ndarray] e
    '''
    #encode a peptide into blosum features
    s=list(seq)
    x = pd.DataFrame([matix[i] for i in seq]).reset_index(drop=True)
    e = x.to_numpy().flatten() 
    # print(x)   
    return e

def blosum50_encode(seq):
    '''
    Encode protein sequence, seq, to one-dimension array.
    Use blosum62 matrix to encode the number.
    input: [string] seq (length = n)
    output: [1x24n ndarray] e
    '''
    #encode a peptide into blosum features
    s=list(seq)
    x = pd.DataFrame([blosum50[i] for i in seq]).reset_index(drop=True)
    e = x.to_numpy().flatten() 
    # print(x)   
    return e

def DecToBinEncode(dec, lowerBound, UpperBound):
    '''Decimal to binary and encode it to len-3 list
    dec: int
        decimal integar, in the range of [lowerBound, UpperBound]
    lowerBound: int
        the lower bound of the permitted decimal
    UpperBound: int
        the upper bound of the permitted decimal

    Return:
    ------
    binCode: int[]
        list of 0 and 1
    '''
    if dec < lowerBound or dec > UpperBound:
        print("decimal out of bound")
        return
    else:
        biList = list(bin(dec)[2:])
        for i in range(len(biList)):
            biList[i] = int(biList[i])
        if len(biList) < 3:
            biList = [0]*(3-len(biList)) + biList
    return biList

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

def r2_score(true, sc):
    return metrics.r2_score(true, sc)

def pearson_score(true, sc):
    r, p = pearsonr(true, sc)
    return r

def find_model(aw):
    '''Find model for alleles of mhcii
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
    fname = os.path.join(os.path.join(model_path, "mhcii"), "BasicMHCII_"+aw+".joblib")
    print(fname)
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return