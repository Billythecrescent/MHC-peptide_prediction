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
from scipy.stats import pearsonr
import epitopepredict as ep
import NullSeq_Functions as NS

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
matrices_path = os.path.join(module_path, "matrices")


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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

def randomPeptideGenerator(TranscribeTableNum, l, seqNum):
    '''Generate random amino acid sequences given a codon table
    TranscribeTableNum: int
        the codon table index according to NCBI
        default: 11
    l: int
        the length of the amino acid sequnce
    seqNum: int
        the number of generated random sequences
    
    Returns
    -------
    AASequences: string[]
        random amino acid sequence list
    '''
    
    AAfile = os.path.join(current_path, "AAUsage.csv")
    N = TranscribeTableNum
    if AAfile is not None:
        if AAfile.split('.')[-1] == 'csv':
            AAUsage = NS.df_to_dict(NS.AAUsage_from_csv(AAfile), N)
            length = l
            AASequence = None
            operatingmode = 'AA Usage Frequency'
        else:
            AASequence = NS.parse_fastafile(AAfile)
            AAUsage = NS.get_AA_Freq(AASequence, N, nucleotide=False)
            operatingmode = 'Existing Sequence - AA'
            if l == None:
                length = len(AASequence)-1
            else:
                length = l
            if ES:
                pass
            else:
                AASequence = None
    AASequences = []
    for i in range(seqNum):
        AASequences.append(NS.get_Random_AA_Seq(AAUsage, length))
    # print(AASequences)
    return AASequences

# for i in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21):
#     randomPeptides = np.array(list(zip(randomPeptideGenerator(11, i, 20), [i]*20)))
#     print(randomPeptides)
#     randomPeptides_df = pd.DataFrame(randomPeptides, columns=['peptide', 'length'])
#     print(randomPeptides_df)
#     randomPeptides_df.to_csv(os.path.join(data_path, "mhcii_random.csv"), mode='a')

def geo_mean(iterable):
    nplist = np.array(iterable)
    for i in range(len(nplist)):
        if nplist[i] < 0 or nplist[i] == 0:
            nplist[i] = 0.001
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

def GenerateDataForCurveDict(method, labels, predicted_scores):
    '''generate specific data format (dictionary) for roc curve
    method: string
        the prediction method name.
    labels: numpy.array
        the array containing the labels
    predicted_scores: np.array
        the array containing the predicted scores of peptides

    return:
    -------
    data_for_curve_dict: dictionary
        key: method
    '''
    if type(labels) != np.ndarray:
        labels = np.array(labels)
    if type(predicted_scores) != np.ndarray:
        predicted_scores = np.array(predicted_scores)
    
    data_for_curve_dict = {method: {}}
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, predicted_scores)
    result_auc = metrics.auc(fpr, tpr)
    data_for_curve_dict[method]['x_axis_item'] = fpr.tolist()
    data_for_curve_dict[method]['y_axis_item'] = tpr.tolist()
    data_for_curve_dict[method]['result_auc'] = result_auc

    return data_for_curve_dict

def test_GenerateDataForCurveDict():
    file_path = os.path.join(current_path, "mhci1_Tumor_result.csv")
    dataset = pd.read_csv(file_path)
    method = "mhci1"
    labels = dataset.binder
    predicted_scores = dataset.MHCi1_log50k
    # print(GenerateDataForCurveDict(method, labels, predicted_scores))

# test_GenerateDataForCurveDict()