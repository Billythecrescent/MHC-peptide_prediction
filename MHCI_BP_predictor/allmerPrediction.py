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
# from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
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
    # print(X)  # 24*9+(4+3+3) = 226

    return X, seq_list

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
    seq_df = pd.DataFrame(seq_list, columns = ['peptide'])
    print(seq_df)
    seqX = seq_df.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)
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
    
    return trueX

# ##--- Test ---###
# reg = MLPRegressor(hidden_layer_sizes=(5), alpha=0.01, max_iter=500,
#                         activation='relu', solver='lbfgs', random_state=2)
# randomPep = PF.randomPeptideGenerator(11, 9, 1)
# allele = "HLA-A*01:01"
# iniY = [0.1]
# iniX, seq_list = AllmerEncoder(allele, randomPep[0], blosum_encode)
# reg.fit(iniX, iniY)
# AllmerPrepredict(allele, "ASFCGSPY", blosum_encode, reg, False)

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
    if len(dataset)<200:
        return
    
    ##initialize the predictor
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=500,
                        activation='relu', solver='lbfgs', random_state=2)
    #create random X and Y as the data for regression initialization
    randomPep = PF.randomPeptideGenerator(11, 9, 1)
    iniX, seq_list = AllmerEncoder(allele, randomPep[0], blosum_encode)
    iniY = [0.1]
    reg.fit(iniX, iniY)

    ##encode the peptide
    X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(allele, x, blosum_encode, reg, False)),1)
    y = dataset.log50k

    ##cross_validation not done
    PreCirNum = 6
    ferror = os.path.join(current_path, "randomStart_neg_mean_squared_error.csv")
    fauc = os.path.join(current_path, "randomStart_roc_auc.csv")
    fr2 = os.path.join(current_path, "randomStart_r2.csv")
    square_error_list = []
    auc_list = []
    r2_list = []
    for i in range(PreCirNum):
        cv_results = cross_validate(reg, X, y, cv=5, scoring = ('neg_mean_squared_error', 'r2'), return_estimator=True)
        square_error_list.append(cv_results['test_neg_mean_squared_error'])
        # auc_list.append(cv_results['test_roc_auc'])
        r2_list.append(cv_results['test_r2'])
        reg = cv_results['estimator']
        print(cv_results.keys())
    # auc_df = pd.DataFrame(auc_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])
    square_error_df = pd.DataFrame(square_error_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])
    r2_df = pd.DataFrame(r2_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])

    # auc_df.to_csv(fauc)
    square_error_df.to_csv(ferror)
    r2_df.to_csv(fr2)
    
    return reg, r2_df#, auc_df

# allele = "Patr-A*0101"
# data_path = os.path.join(data_path, "modified_mhc.20130222.csv")
# dataset = pd.read_csv(data_path)
# shuffled_dataset = shuffle(dataset, random_state=0)
# allele_dataset = shuffled_dataset.loc[shuffled_dataset['allele'] == allele]
# # print(allele_dataset)
# # print(dataset)
# hidden_node = 5
# RandomStartPredictor(dataset, allele, blosum_encode, hidden_node)

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
    reg = PF.find_model(aw, 9)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return 

    ##encode the peptide
    X = dataset.peptide.apply(lambda x: pd.Series(AllmerPrepredict(allele, x, blosum_encode, reg, True)),1)
    y = dataset.log50k

    ##cross_validation
    PreCirNum = 3
    ferror = os.path.join(current_path, "existStart_neg_mean_squared_error.csv")
    fauc = os.path.join(current_path, "existStart_roc_auc.csv")
    fr2 = os.path.join(current_path, "existStart_r2.csv")
    square_error_list = []
    auc_list = []
    r2_list = []
    for i in range(PreCirNum):
        cv_results = cross_validate(reg, X, y, cv=5, scoring = ('neg_mean_squared_error', 'r2'), return_estimator=True)
        square_error_list.append(cv_results['test_neg_mean_squared_error'])
        # auc_list.append(cv_results['test_roc_auc'])
        r2_list.append(cv_results['test_r2'])
        reg = cv_results['estimator']
        print(cv_results.keys())
    # auc_df = pd.DataFrame(auc_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])
    square_error_df = pd.DataFrame(square_error_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])
    r2_df = pd.DataFrame(r2_list, columns = [str(i)+"-fold" for i in range(1, PreCirNum+1)])

    # auc_df.to_csv(fauc)
    square_error_df.to_csv(ferror)
    r2_df.to_csv(fr2)

    return reg, r2_df#, auc_df

# allele = "H-2-Kd"
# data_path = os.path.join(data_path, "modified_mhc.20130222.csv")
# dataset = pd.read_csv(data_path)
# shuffled_dataset = shuffle(dataset, random_state=0)
# allele_dataset = shuffled_dataset.loc[shuffled_dataset['allele'] == allele]
# # print(allele_dataset)
# # print(dataset)
# hidden_node = 5
# ExistStartPredictor(dataset, allele, blosum_encode, hidden_node)

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
        reg, auc_df = RandomStartPredictor(dataset, allele, blosum_encode, hidden_node)
        startType = "RandomStart"
    else:
        reg, auc_df = ExistStartPredictor(dataset, allele, blosum_encode, hidden_node)
        startType = "ExistStart"

    return reg, auc_df, startType


def BuildPredictor(dataset, hidden_node, ifRandomStart):
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
    
    Return:
    ------
    None
    '''
    #shuffle dataset
    shuffled_dataset = shuffle(dataset, random_state=0)
    # print(shuffled_dataset)
    alleles = shuffled_dataset.allele.unique().tolist()

    path = os.path.join(model_path, "allmer")
    alleles_auc = []
    for allele in alleles:
        ##cross validation, determing the training and testing data
        #Here I need to get the score
        reg, auc_df, startType = allmerPredictor(dataset, allele, blosum_encode, hidden_node, ifRandomStart)
        alleles_auc.append(auc_df)
        aw = re.sub('[*:]','_',allele)
        fname = os.path.join(os.path.join(path, startType), aw +'.joblib')
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)
            print("%s fitting of allele %s is done" %(startType, allele))
    

# data_path = os.path.join(data_path, "mhci.20130222.csv")
# dataset = pd.read_csv(data_path)
# BuildPredictor(dataset)