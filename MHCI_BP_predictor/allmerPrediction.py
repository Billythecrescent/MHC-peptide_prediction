'''
File: allmerPrediction.py
Author: Mucun Hou
Date: Apr 10, 2020
Description: This script is the update version of Non9mer_Predictor, as in
    NetMHC4.0, uses allmer data (including 9mer, 8mer, 10mer, 11mer etc.).
    Use cross validation for evaluation.
'''

import os, re
import pandas as pd
import numpy as np
from time import time
# from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import *
import joblib
import epitopepredict as ep

import PredictionFunction as PF

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

blosum_encode = PF.blosum_encode

# print(type(blosum_encode("ASFCGSPY")), blosum_encode("ASFCGSPY").shape)

def SlideTo9mer(seq):
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
    seq_list = []
    length_pep_list = []
    inserAdele_len_list = []
    inserAdele_pos_list = []
    if seqLen <= 8:
        inseLen = affiCoreLen - seqLen
        #turn 8mer to 9mer
        for i in range(seqLen+1):
            seq_list.append((seq[:i] + 'X'*inseLen + seq[i:]))
            length_pep_list.append([1, 0, 0, 0])
            inserAdele_len_list.append([1, 1, 1] if inseLen > 6 else PF.DecToBinEncode(inseLen, 0, 6))
            inserAdele_pos_list.append([1, 0, 0] if i == 0 else ([0, 0, 1] if i == seqLen else [0, 1, 0]))

    elif seqLen == 9:
        seq_list.append(seq)
        length_pep_list.append([0, 1, 0, 0])
        inserAdele_len_list.append([0, 0, 0])
        inserAdele_pos_list.append([0, 0, 0])

    elif seqLen >= 10:
        deleLen = seqLen - affiCoreLen
        #turen 10mer to 9mer
        for i in range(affiCoreLen+1):
            seq_list.append(seq[:i] + seq[(i+deleLen):])
            if seqLen == 10:
                length_pep_list.append([0, 0, 1, 0])
            else:
                length_pep_list.append([0, 0, 0, 1])
            inserAdele_len_list.append([1, 1, 1] if deleLen > 6 else PF.DecToBinEncode(deleLen, 0, 6))
            inserAdele_pos_list.append([1, 0, 0] if i == 0 else ([0, 0, 1] if i == affiCoreLen else [0, 1, 0]))
    #combine the columns to form a dataframe
    combinded_list = []
    if len(seq_list) == len(length_pep_list) == len(inserAdele_len_list) ==len(inserAdele_pos_list):
        for i in range(len(seq_list)):
            combinded_list.append(length_pep_list[i] + inserAdele_len_list[i] + inserAdele_pos_list[i])
    else:
        print("the dimension of the encoding list is not consistent")
        return 

    return seq_list, combinded_list

# SlideTo9mer("ABCDEFGH")

def SeqEncoding(seq, feature, blosum_encode):
    '''encode the sequence by blosum_encode and combine it with the feature
        of peptide length, insertion/deletion length&position
    seq_list: string
        the 9mer peptides generated from allmer by 'SlideTo9mer'
    feature: int[]
        the encoding of the feartures, described above
    blosum_encode: function
        the encoding function for seq
    
    Return:
    -------
    encoded: np.array
        the encoded sequence with its features
    '''
    encoded_peptide = blosum_encode(seq)
    encoded = np.hstack((encoded_peptide, np.array(feature)))

    return encoded

# seq_list, feature_list = SlideTo9mer("ABCDEFGH")
# print(SeqEncoding(seq_list[0], feature_list[0], blosum_encode))

def AllmerEncoder(allele, seq, blosum_encode):
    '''Encode the peptide of allmer, with the encoding of peptide length, 
        deletion/insertion length, and deletion/insertion position
    allele: string 
        the name of the allele
    seq: string
        the sequence of the peptide, with the length of 8, 9, 10, 11 or other
    blosum_encode: string
        the name of the blosum_encode function

    Return:
    -------
    encoding: numpy.array
        the encoding array of seq, X
    seq_list: string[]
        the list of the potential 9mer binding core
    '''

    seq_list, feature_list = SlideTo9mer(seq)
    seq_df = pd.DataFrame(seq_list, columns=['peptide'])
    X = seq_df.peptide.apply(lambda x: pd.Series(SeqEncoding(x, feature_list[seq_list.index(x)], blosum_encode)),1)
    #split X
    splitX = np.split(X, [216], axis=1)
    #scale the data to [-1, 1] use Abs
    scaler = MaxAbsScaler()
    # print(pd.DataFrame(scaler.fit_transform(splitX[0]), columns = [i for i in range(216)]))
    X = pd.concat((pd.DataFrame(scaler.fit_transform(splitX[0]), columns = [i for i in range(216)]), splitX[1]), axis=1)
    # print(X)  # 24*9+(4+3+3) = 226

    return X, seq_list

# AllmerEncoder("HLA-A*01:01", "ABCDEFGH", blosum_encode)

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
    X, seq_list = AllmerEncoder(allele, seq, blosum_encode)

    seqX = pd.DataFrame(np.split(X, [216], axis=1)[0], columns = [i for i in range(216)])
    #predict
    if state == True:
        scores = reg.predict(seqX).tolist()
    else:
        scores = reg.predict(X).tolist()
    # print(scores)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    binding_core = seq_list[max_score_index]
    # print(binding_core, max_score)
    
    trueX = X.iloc[max_score_index].to_numpy()
    # print(trueX, trueX.shape)
    # print(seq+"_"+"done", len(seq))
    return trueX

def test_AllmerPrepredict():
    ##--- Test ---###
    reg = MLPRegressor(hidden_layer_sizes=(5), alpha=0.01, max_iter=500,
                            activation='relu', solver='adam', random_state=2)
    randomPep = PF.randomPeptideGenerator(11, 9, 1)
    allele = "HLA-A*01:01"
    iniY = [0.1]
    iniX, seq_list = AllmerEncoder(allele, randomPep[0], blosum_encode)
    reg.fit(iniX, iniY)
    AllmerPrepredict(allele, "ASFCGSPY", blosum_encode, reg, False)

def RandomStartPredictor(dataset, allele, blosum_encode, hidden_node):
    '''Prediction of specific allele with initial random-set predictor, 
        perform cross validation
    dataset: Dataframe
        MUST has 'allele', 'peptide', 'log50k', 'length' columns
    allele: string
        the name of the allele
    blosum_encode: string
        the name of the blosum_encode function
    hidden_node: int
        the number of hidden layer nodes

    Return:
    -------
    reg: regression predictor
        The trained predictor
    auc_df: DataFrame
        The auc value of all prediction circles
    '''
    # if len(dataset)<200:
    #     return
    
    ##initialize the predictor
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    #create random X and Y as the data for regression initialization
    randomPep = PF.randomPeptideGenerator(11, 9, 1)
    iniX, seq_list = AllmerEncoder(allele, randomPep[0], blosum_encode)
    iniY = [0.1]
    reg.fit(iniX, iniY)

    y = dataset.log50k.to_numpy()

    #X is encoded below

    ##cross_validation not done
    PreCirNum = 10
    fauc = os.path.join(current_path, "allmer_randomStart_roc_auc.csv")
    fr = os.path.join(current_path, "allmer_randomStart_pearson.csv")
    avg_auc_list = []
    avg_r_list = []
    for i in range(PreCirNum):
        print("allele %s round %d starts" % (allele, i))
        auc_list = []
        r_list = []
        ##encode the peptide
        X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(allele, x, blosum_encode, reg, False)),1).to_numpy()
        # print(X)
        kf = KFold(n_splits=5, shuffle=True)
        for k, (train, test) in enumerate(kf.split(X, y)):
            print("allele %s round %d fold %d starts" %(allele, i, k))
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            r = PF.pearson_score(y[test], scores)
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

def test_RandomStart():
    allele = "Patr-A*0101"
    # allele = "H-2-Kd"
    data_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(data_path)
    # shuffled_dataset = shuffle(dataset, random_state=0)
    allele_dataset = dataset.loc[dataset['allele'] == allele]
    # print(allele_dataset)
    hidden_node = 5
    RandomStartPredictor(allele_dataset, allele, blosum_encode, hidden_node)

def ExistStartPredictor(dataset, allele, blosum_encode, hidden_node):
    '''Prediction of specific allele with initial exist-set predictor, 
        perform cross validation
    dataset: Dataframe
        MUST has 'allele', 'peptide', 'log50k', 'length' columns
    allele: string
        the name of the allele
    blosum_encode: string
        the name of the blosum_encode function
    hidden_node: int
        the number of hidden layer nodes

    Return:
    -------
    reg: regression predictor
        The trained predictor
    auc_df: DataFrame
        The auc value of the all prediction circles
    '''
    if len(dataset)<200:
        return
    
    ##find the corresponding predictor
    aw = re.sub('[*:]','_',allele)
    exist_reg = PF.find_model(aw, 9)
    if exist_reg is None:
        print ('Locals do not have model for this allele.')
        return 
    
    #create new regression
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=5000, early_stopping=True,
                        activation='relu', solver='adam', random_state=2)

    ##encode the peptide
    y = dataset.log50k.to_numpy()
    #X is encoded below

    ##cross_validation
    PreCirNum = 10
    fauc = os.path.join(current_path, "allmer_ExistStart_roc_auc.csv")
    fr = os.path.join(current_path, "allmer_ExistStart_pearson.csv")
    avg_auc_list = []
    avg_r_list = []
    for i in range(PreCirNum):
        print("Round %d starts" % i)
        auc_list = []
        r_list = []
        ##encode the peptide
        if i == 0:
            X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(allele, x, blosum_encode, exist_reg, True)),1).to_numpy()
        else:
            X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(allele, x, blosum_encode, reg, False)),1).to_numpy()
        # print(X)
        kf = KFold(n_splits=5, shuffle=True)
        for k, (train, test) in enumerate(kf.split(X, y)):
            print("allele %s round %d fold %d starts" %(allele, i, k))
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            auc_list.append(auc)
            r = PF.pearson_score(y[test], scores)
            r_list.append(r)

        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        if len(avg_auc_list) > 0 and avg_auc < 0.995*avg_auc_list[-1][0]:
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


def test_ExistStart():
    allele = "H-2-Kb"
    data_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(data_path)
    allele_dataset = dataset.loc[dataset['allele'] == allele]
    # print(allele_dataset)
    # print(dataset)
    hidden_node = 5
    ExistStartPredictor(allele_dataset, allele, blosum_encode, hidden_node)

def allmerPredictor(dataset, allele, blosum_encode, hidden_node, ifRandomStart):
    '''Choose prediction strategy according to ifRandomStart and perform cross validation
        random start uses initialized regression predictor to iterate fitting
        exist start uses existed model to prepredict the binding core of the 
            peptide and other features
    dataset: Dataframe
        MUST has 'allele', 'peptide', 'log50k', 'length' columns
    allele: string
        the name of the allele
    blosum_encode: string
        the name of the blosum_encode function
    hidden_node: int
        the number of hidden layer nodes
    ifRandomStart: Boolean
        Whether it is random start or exist start
    '''
    if ifRandomStart:
        reg, auc_df, r_df = RandomStartPredictor(dataset, allele, blosum_encode, hidden_node)
    else:
        reg, auc_df, r_df = ExistStartPredictor(dataset, allele, blosum_encode, hidden_node)

    return reg, auc_df, r_df


def BuildPredictor(dataset, hidden_node, ifRandomStart, score_filename):
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
    
    for allele in alleles:
        ##cross validation, determing the training and testing data
        #Here I need to get the score
        allele_dataset = dataset.loc[dataset['allele'] == allele]
        reg, auc_df, r_df= allmerPredictor(allele_dataset, allele, blosum_encode, hidden_node, ifRandomStart)
        
        auc_df.to_csv(os.path.join(current_path, score_filename + "_auc.csv"), mode='a', header=False)
        r_df.to_csv(os.path.join(current_path, score_filename + "_PCC.csv"), mode='a', header=False)
        
        # aw = re.sub('[*:]','_',allele)
        # fname = os.path.join(os.path.join(path, StartType), aw +'.joblib')
        # if reg is not None:
        #     joblib.dump(reg, fname, protocol=2)
        #     print("%s fitting of allele %s is done" %(StartType, allele))

def main():
    t0 = time()

    path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(path)
    for i in range(7, 11):
        BuildPredictor(dataset, i, False, "AllmerPredictionResult")
    
    t1 = time()
    print ("Elapsed time (m):", (t1-t0)/60)

main()