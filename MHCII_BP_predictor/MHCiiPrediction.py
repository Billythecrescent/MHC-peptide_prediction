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

blosum_encode = PF2.blosum50_encode

def EncodeTo9mer(seq, blosum_encode):
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
                        RightPfr = np.concatenate((blosum_encode(seq[(slideLeft+slideLen):]), np.zeros((3+slideLeft-slideEnd)*24)), axis = 0)
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
            # print(encoded.shape)
            encoded_list.append(encoded)
    else:
        print("the dimension of the encoding list is not consistent")
        return 
    # print(encoded_list[0], encoded_list[0].shape) #shape=(367,)
    # print(len(encoded_list))
    return encoded_list

# EncodeTo9mer("CELGEWVFSSVQPPK", blosum_encode)

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
    ModelPath = os.path.join(model_path, "mhcii")
    aw = re.sub('[*:]','_', allele) 
    fname = os.path.join(ModelPath, "BasicMHCII_"+aw+".joblib")
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

# test_Basic9merPrediction()

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

def AllmerPrepredict(seq, blosum_encode, reg):
    '''Preprediction of the peptide, to find out the binding core and encode the peptide
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
    encoded = EncodeTo9mer(seq, blosum_encode)
    scores = [reg.predict(encoded[i].reshape(1,-1)) for i in range(len(encoded))]
    # print(scores)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    
    # print(max_score)
    
    trueX = encoded[max_score_index]
    # print(trueX, trueX.shape)
    # print(seq+"_"+"done", len(seq))
    return trueX

def test_AllmerPrepredict():
    peptide = "CELGEWVFSSVQPPK"
    allele = "HLA-DRB1*0101"
    aw = re.sub('[*:]','_', allele) 
    reg = PF2.find_model(aw)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return
    X = AllmerPrepredict(peptide, blosum_encode, reg)
    print(X, X.shape)

# test_AllmerPrepredict()

def MHCiiPredictor(allele, dataset, hidden_node, blosum_encode):
    y = dataset.log50k.to_numpy()

    aw = re.sub('[*:]','_', allele) 
    ExistReg = PF2.find_model(aw)
    if ExistReg is None:
        print ('Locals do not have initial model for this allele.')
        return

    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    ##cross_validation not done
    PreCirNum = 10
    fauc = os.path.join(current_path, "MHCii-"+aw+"_roc_auc.csv")
    fr = os.path.join(current_path, "MHCpan-"+aw+"_pearson.csv")
    avg_auc_list = []
    avg_r_list = []
    for rd in range(PreCirNum):
        print("Allele %s round %d starts" % (allele, rd))
        auc_list = []
        r_list = []
        ##encode the peptide
        if rd == 0:
            X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(x, blosum_encode, ExistReg)),1).to_numpy()
        else:
            X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(x, blosum_encode, reg)),1).to_numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for k, (train, test) in enumerate(kf.split(X, y)):
            print("Round %d fold %d starts" %(rd, k))
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            # print(scores)
            auc = PF2.auc_score(y[test], scores, cutoff=.426)
            r = PF2.pearson_score(y[test], scores)
            auc_list.append(auc)
            r_list.append(r)
        
        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        if len(avg_auc_list) > 0 and avg_auc < 0.99*avg_auc_list[-1][0]:
            break
        avg_auc_list.append(np.array([avg_auc]+auc_list))
        avg_r_list.append(np.array([avg_r]+r_list))

    avg_auc_list = np.array(avg_auc_list)
    avg_r_list = np.array(avg_r_list)

    # print(avg_auc_list)
    # print(avg_r_list)
    
    auc_df = pd.DataFrame(np.array(avg_auc_list[-1]).reshape(1,-1), columns = ['avg_AUC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    r_df = pd.DataFrame(np.array(avg_r_list[-1]).reshape(1,-1), columns = ['avg_PCC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    print(auc_df)
    print(r_df)

    # auc_df.to_csv(fauc)
    # r_df.to_csv(fr)
    
    return reg, auc_df, r_df

def BuildPredictor(dataset, hidden_node, score_filename):
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

    print("Prediction\nEncode Method: blosum50\nhidden node: %d\noutput filename: %s\n" \
     %(hidden_node, score_filename))

    for allele in alleles:
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        aw = re.sub('[*:]','_', allele)
        reg, auc_df, r_df= MHCiiPredictor(allele, allele_dataset, hidden_node, blosum_encode)
        auc_df.to_csv(os.path.join(current_path, score_filename + '_' + aw + "_auc.csv"), mode='a', header=False)
        r_df.to_csv(os.path.join(current_path, score_filename + '_' + aw + "_PCC.csv"), mode='a', header=False)

        #store the predictor
        ModelPath = os.path.join(model_path, "mhcii")
        aw = re.sub('[*:]','_', allele) 
        fname = os.path.join(ModelPath, "MHCII_"+aw+".joblib")
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("MHCpan predictor for %s is done." %allele)
            print("Model path: %s" %fname)

def AffinityPredict(dataset, outputFile=None):
    alleles = dataset.allele.unique().tolist()
    df_list = []
    # print(dataset)
    for allele in alleles:
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        
        aw = re.sub('[*:]','_',allele) 
        ModelPath = os.path.join(model_path, "mhcii")
        fname = os.path.join(ModelPath, "MHCII_"+aw+".joblib")
        if os.path.exists(fname):
            print(fname)
            reg = joblib.load(fname)
        else:
            print("Can not find the model for %s in %s" %(allele, fname))
            continue

        X = allele_dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(x, blosum_encode, reg)),1).to_numpy()
        scores = pd.DataFrame(reg.predict(X), columns=['log50k'], index=allele_dataset.index)
        result = pd.concat([allele_dataset, scores], axis=1)
        # print(result)
        df_list.append(result)    

    combined_df = pd.concat(df_list, axis=0, sort=True)
    combined_df.sort_index(inplace=True)
    print(combined_df)

    if outputFile != None:
        combined_df.to_csv(outputFile)

def testAffinityPredict():
    # path = os.path.join(data_path, "mhcii_random.csv")
    path = os.path.join(data_path, "mhciiTumor_dataset.csv")
    # path = os.path.join(data_path, "modified_mhciTumor_dataset.csv")
    dataset = pd.read_csv(path)
    # AffinityPredict(dataset, False, os.path.join(current_path, "mhci3_ExistStart_Tumor_result.csv"))
    # AffinityPredict(dataset, os.path.join(current_path, "mhcii_random_result.csv"))
    AffinityPredict(dataset, os.path.join(current_path, "mhcii_Tumor_result.csv"))

testAffinityPredict()

def main():
    t0 = time()

    path = os.path.join(data_path, "mhcii-DRB1-dataset.csv")
    dataset = pd.read_csv(path)
    dataset = dataset.loc[dataset['allele'] == "HLA-DRB1*0101"]
    dataset = shuffle(dataset, random_state=0)
    # print(dataset.shape)
    hidden_node = 20
    BuildPredictor(dataset, hidden_node, "mhciiPredictionResult")

    t1 = time()
    print ("Elapsed time (m):", (t1-t0)/60)

# main()