'''
File: MHCpanPrediction.py
Author: Mucun Hou
Date: Apr 18, 2020
Description: This script is to build and evaluate the prediction ANN model
    for mhcii-peptide binding.
'''

import os, re
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from time import time
# from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import *
import joblib
import epitopepredict as ep

import PredictionFunction_mhcii as PF2

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
mhcii_path = os.path.join(data_path, "mhcii")

blosum_encode = PF2.blosum_encode

def EncodeTo9mer(seq):
    '''Transform allmer sequence to potential 9mer binding core
        as described in NetMHC4.0, different from NetMHC3.0 L-mer approximation
    seq: string
        the sequence of peptide, length of which is 8, 9, 10, 11, or other.

    Return:
    -------
    seq_list: string[]
        the list of potential binding core
    combined_list: list[int[]]
        the list of encoding nodes without the peptide itself
        encoding note: like [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            first 4 nodes represent peptide length, middle 3 represent insertion/deletion
            length, last 3 represent insertion/deletion position
    '''
    affiCoreLen = 9
    seqLen = len(seq)
    seqLenEncode = 1/(1+np.exp((seqLen-15)/2))

    seq_list = []               #core sequence
    length_slide_list = []      #slide length
    pos_slide_list = []         #slide position    
    pfr_list = []               #franking sequence
    
    if seqLen == 8:
        print("invalid length of mhcii peptide: 8")
        return 

    elif seqLen == 9:
        seq_list.append(blosum_encode(seq))
        length_slide_list.append(np.array([0, 0]))
        pos_slide_list.append(np.array([0, 0, 0]))
        pfr_list.append(np.zeros(6*24))

    elif seqLen > 9:
        #slide creation
        for slideLen in range(9, seqLen+1):
            bulgLen = slideLen - affiCoreLen
            slideEnd = seqLen-slideLen
            #slide start
            for slideLeft in range(slideEnd+1):
                slideLenEncode = (slideLen-9)/seqLen
                slidePos = [1, 0, 0] if slideLeft == 0 else ([0, 0, 1] if slideLeft == (slideEnd) else [0, 1, 0])
                slidePos = np.array(slidePos)
                LeftPfr = np.zeros(3*24)
                RightPfr = np.zeros(3*24)
                if slideLeft != 0:
                    if slideLeft < 3:
                        LeftPfr = np.concatenate((np.zeros((3-slideLeft)*24), blosum_encode(seq[:slideLeft])), axis = 0)
                    else:
                        LeftPfr = blosum_encode(seq[(slideLeft-3):slideLeft])
                if slideLeft != (slideEnd):
                    if slideLeft > (slideEnd-3):
                        RightPfr = np.concatenate((blosum_encode(seq[slideEnd-slideLeft]), np.zeros((3+slideLeft-slideEnd)*24)), axis = 0)
                    else:
                        RightPfr = blosum_encode(seq[(slideLeft+slideLen):(slideLeft+slideLen+3)])
                # print(LeftPfr, LeftPfr.shape)
                # print(RightPfr, RightPfr.shape)
                Pfr = np.concatenate((LeftPfr, RightPfr), axis = 0)
                # print(Pfr, Pfr.shape)
                
                if slideLen == 9:
                    seqEncode = blosum_encode(seq[slideLeft:(slideLeft+slideLen)])
                    seq_list.append(seqEncode)
                    length_slide_list.append(np.array([slideLenEncode, 1-slideLenEncode]))
                    pos_slide_list.append(slidePos)
                    pfr_list.append(Pfr)
                    continue

                #bulging starts
                for i in range(1, affiCoreLen):
                    slideSeq = seq[slideLeft:(slideLeft+slideLen)]
                    seqEncode = blosum_encode(slideSeq[:i] + slideSeq[(i+bulgLen):])
                    seq_list.append(seqEncode)
                    length_slide_list.append(np.array([slideLenEncode, 1-slideLenEncode]))
                    pos_slide_list.append(slidePos)
                    pfr_list.append(Pfr)
    
    encoded_list = []
    if len(seq_list) == len(length_slide_list) == len(pos_slide_list) ==len(pfr_list):
        length_seq_list = [np.array([seqLenEncode, 1-seqLenEncode])]*len(seq_list)
        # print(length_seq_list[0],length_seq_list[0].shape)
        # print(seq_list[0],seq_list[0].shape)
        for i in range(len(seq_list)):
            encoded = np.concatenate((seq_list[i], pfr_list[i], length_seq_list[i], length_slide_list[i], pos_slide_list[i]), axis=0)
            encoded_list.append(encoded)
    else:
        print("the dimension of the encoding list is not consistent")
        return 
    print(encoded_list[0], encoded_list[0].shape) #shape=(367,)
    print(encoded_list)
    return encoded_list

# EncodeTo9mer("CELGEWVFS")

def NinerEncode(seq):
    '''Encode 9mer mhcii peptide, ready for baseline prediction (use only 9mer for a simple model)
    seq: string
        9mer peptide sequence to be encoded

    Return:
    -------
    encoded: numpy.array
        encoded sequence, must be (367,)
    '''
    seqLen = len(seq)
    seqLenEncode = np.array([1/(1+np.exp((seqLen-15)/2)), 1-1/(1+np.exp((seqLen-15)/2))])
    seqEncode = blosum_encode(seq)
    length_slide =np.array([0, 0])
    pos_slide = np.array([0, 0, 0])
    pfr = np.zeros(6*24)
    encoded = np.concatenate((seqEncode, pfr, seqLenEncode, length_slide, pos_slide), axis=0)

    return encoded

# NinerEncode("CELGEWVFS")

def Basic9merPrediction(allele, dataset, hidden_node, blosum_encode):
    y = dataset.log50k.to_numpy()

    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    X = dataset.peptide.apply(lambda x: pd.Series(NinerEncode(x)),1).to_numpy()

    reg.fit(X,y) 

    #store the predictor
    PanModelPath = os.path.join(model_path, "mhcii")
    aw = re.sub('[*:]','_', allele) 
    fname = os.path.join(PanModelPath, "BasicMHCII_"+aw+".joblib")
    if reg is not None:
        joblib.dump(reg, fname, protocol=2)
        print("basic MHCpan predictor is done.")
        print("Model path: %s" %fname)

def test_Basic9merPrediction():
    allele = "HLA-DRB1*0101"
    path = os.path.join(data_path, "mhcii-DRB1-dataset.csv")
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['length'] == 9]
    dataset = dataset.loc[dataset['allele'] == allele]
    dataset = shuffle(dataset, random_state=0)
    hidden_node = 10
    Basic9merPrediction(allele, dataset, hidden_node, blosum_encode)

test_Basic9merPrediction()

def Basic9merCrossValid(dataset, hidden_node, blosum_encode):
    y = dataset.log50k.to_numpy()

    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    X = dataset.peptide.apply(lambda x: pd.Series(NinerEncode(x)),1).to_numpy()

    auc_list = []
    r_list = []
    kf = KFold(n_splits=5, shuffle=True)
    for k, (train, test) in enumerate(kf.split(X, y)):
        print("Hidden nodee:%d, fold %d starts" %(hidden_node, k))
        t0 = time()
        reg.fit(X[train], y[train])
        scores = reg.predict(X[test])
        auc = PF2.auc_score(y[test], scores, cutoff=.426)
        r = PF2.pearson_score(y[test], scores)
        auc_list.append(auc)
        r_list.append(r)
        t1 = time()
        print("fold %d done, run in Elapsed time %d(m)" %(k, (t1-t0)/60))
    print(auc_list)
    print(r_list)
    avg_auc = np.mean(auc_list)
    avg_r = np.mean(r_list)

    return avg_auc, avg_r

def test_Basic9merCrossValid():
    path = os.path.join(data_path, "mhcii-DRB1-dataset.csv")
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['length'] == 9]
    dataset = dataset.loc[dataset['allele'] == "HLA-DRB1*0101"]
    dataset = shuffle(dataset, random_state=0)
    # print(dataset.shape)
    AUClist = []
    # PCClist = []
    hidden_range = range(1, 21)
    for hidden_node in hidden_range:
        auc_list = []
        # r_list = []
        for j in range(20):
            auc, r = Basic9merCrossValid(dataset, hidden_node, blosum_encode)
            auc_list.append(auc)
            # r_list.append(r)
        AUClist.append(np.mean(auc_list))
        # PCClist.append(np.mean(r_list))
    Scoredf = pd.DataFrame(AUClist, index=[i for i in hidden_range], columns = ["AUC"])
    Scoredf.to_csv(os.path.join(current_path, "basicMHCii_crossValidation.csv"))
    print(Scoredf)

# test_Basic9merCrossValid()

def AllmerPrepredict(allele, seq, blosum_encode, reg, state):
    '''Preprediction of the peptide, to find out the binding core and encode the peptide
    allele: string 
        the name of the allele
    seq: string
        the sequence of the peptide, with the length of 8, 9, 10, 11 or other
    blosum_encode: string
        the name of the blosum_encode function
    reg: regression predictor
        the initialized regression predictor
    state: boolean
        the indicator of whether it is random start of exist start
        True if exist-start
        False if random-start

    Return:
    -------
    trueX: numpy.ndarray
        the only true encoding of the sequence
    '''
    encoded = EncodeTo9mer(allele, seq, blosum_encode)
    scores = [reg.predict(encoded[i]) for i in range(len(encoded))]

    # print(scores)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    
    # print(max_score)
    
    trueX = encoded[max_score_index]
    # print(trueX, trueX.shape)
    # print(seq+"_"+"done", len(seq))
    return trueX

""" def BuildPredictor(dataset, hidden_node, ifRandomStart, score_filename):
    '''Build predictor according to whether it is random start or exist start
        random start uses initialized regression predictor to iterate fitting
        exist start uses existed model to prepredict the binding core of the 
            peptide and other features
    dataset: DataFrame
        the dataset of different alleles
    hidden_node: int
        the number of hidden layer nodes
    ifRandomStart: Boolean
        Whether it is random start or exist start
    score_filename: string 
        the name of the output auc file (auc, PCC)
    
    Return:
    ------
    None
    '''
    #shuffle dataset
    # shuffled_dataset = shuffle(dataset, random_state=0)
    # print(shuffled_dataset)
    alleles = dataset.allele.unique().tolist()

    StartType = 'RandomStart'
    if ifRandomStart == False:
        StartType = 'ExistStart'

    print("Prediction Mode: %s Prediction\nEncode Method: blosum62\nhidden node: %d\noutput filename: %s\n" \
     %(StartType, hidden_node, score_filename))


    path = os.path.join(model_path, "allmer")
    alleles_auc = []
    alleles_PCC = []
    for allele in alleles:
        ##cross validation, determing the training and testing data
        #Here I need to get the score
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        reg, auc_df, r_df= allmerPredictor(allele_dataset, allele, blosum_encode, hidden_node, ifRandomStart)
        auc_df.to_csv(os.path.join(current_path, score_filename + "_auc.csv"), mode='a', header=False)
        r_df.to_csv(os.path.join(current_path, score_filename + "_PCC.csv"), mode='a', header=False)
        alleles_auc.append(auc_df)
        alleles_PCC.append(r_df)
        # aw = re.sub('[*:]','_',allele)
        # fname = os.path.join(os.path.join(path, startType), aw +'.joblib')
        # if reg is not None:
        #     joblib.dump(reg, fname, protocol=2)
        #     print("%s fitting of allele %s is done" %(startType, allele))
    alleles_auc_df = pd.DataFrame(np.array(alleles_auc).reshape(1,-1))
    alleles_auc_df.to_csv(os.path.join(current_path, score_filename + "_complete_auc.csv"))
    alleles_PCC_df = pd.DataFrame(np.array(alleles_PCC).reshape(1,-1))
    alleles_PCC_df.to_csv(os.path.join(current_path, score_filename + "_complete_PCC.csv"))

def main():
    t0 = time()

    data_path = os.path.join(data_path, "mhcii-DRB1-dataset.csv")
    dataset = pd.read_csv(data_path)
    hidden_node = 20
    BuildPredictor(dataset, hidden_node, False, "AllmerPredictionResult")

    t1 = time()
    print ("Elapsed time (m):", (t1-t0)/60) """