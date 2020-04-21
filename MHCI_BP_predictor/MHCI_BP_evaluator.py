#MHCI_BP_evaluator.py

import os, sys, math, re
import numpy as np
import pandas as pd
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
import epitopepredict as ep

from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor

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

def find_model(allele):
    fname = os.path.join(model_path, allele+'.joblib')
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return

def evaluate_predictor(X, y, allele):

    #print (len(data))
    # print(list(data.peptide), allele)
    reg = find_model(allele)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return
    scores = reg.predict(X)

    #Generate auc value
    auc = auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def predict_8mer(allele, seq):

    #turn 8mer to 9mer
    seq_list = []
    for i in range(5):
        seq_list.append(seq[:(i+3)]+'X'+seq[(i+3):])
    seq_df = pd.DataFrame(seq_list, columns=['peptide'])

    #encode
    X = seq_df.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)

    #find model
    aw = re.sub('[*:]','_',allele) 
    reg = find_model(aw)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return 0.5
    
    #predict
    scores = reg.predict(X)

    #geometric mean
    mean_score = geo_mean(scores)
    return mean_score

# print(predict_8mer("HLA-A*01:01", 'LTDFGLSK'))

def predict_non9mer(allele, seq):

    seq_list = []
    seqLen = len(seq)
    if seqLen == 8:
        #turn 8mer to 9mer
        for i in range(5):
            seq_list.append(seq[:(i+3)]+'X'+seq[(i+3):])

    elif seqLen >= 10:
        bulg = seqLen - 9
        #turen 10mer to 9mer
        for i in range(3, 9):
            seq_list.append(seq[:i]+seq[(i+bulg):])
  
    seq_df = pd.DataFrame(seq_list, columns=['peptide'])
    print(seq_df)

    #encode
    X = seq_df.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)

    #find model
    aw = re.sub('[*:]','_',allele) 
    reg = find_model(aw)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return 0.5
    
    #predict
    scores = reg.predict(X)
    # print(scores)

    #geometric mean
    mean_score = geo_mean(scores)
    return mean_score

# print(predict_non9mer("HLA-A*01:01", "YYRYPTGESY"))
# print(predict_non9mer("HLA-A*01:01", "YSLEYFQFVKK"))
# print(predict_non9mer('HLA-A*01:01', "ABCDEABCDEABCDEABCDEABCDEABCDE"))

def LengthFree_predictor(allele, data):

    y = data.log50k
    
    data_scores = data.peptide.apply(lambda x: predict_non9mer(allele, x))
    data_scores.columns = ['score']
    # print(data_scores)

    #Generate auc value
    auc = auc_score(y,data_scores,cutoff=.426)
    print(auc)

    return auc

## Primary Test ##
# allele = "HLA-A*01:01"
# test_data_8 = ep.get_training_set(allele, length=8)
# test_data_10 = ep.get_training_set(allele, length=10)
# test_data_11 = ep.get_training_set(allele, length=11)
# LengthFree_predictor(allele, test_data_11)

## Secondary Test ##
# dataset_filename = os.path.join(data_path, "evalset_8mer_normalization.csv")
# df = pd.read_csv(dataset_filename)
# alleles = df.allele.unique()
# # print(alleles)
# allele = alleles[12]
# print(allele)
# data = df.loc[df['allele'] == allele]
# result = LengthFree_predictor(allele, data)


## beita Test ##
def Process_LengthFree_Prediction(dataset_filename):
    auc_list = []
    df = pd.read_csv(dataset_filename)
    alleles = df.allele.unique()
    for allele in alleles:
        data = df.loc[df['allele'] == allele]
        result = LengthFree_predictor(allele, data)
        # print(result)
        auc_list.append(result)
    
    print(auc_list)
    # print(auc_list)
    # print(alleles.tolist())
    #Write Result
    auc_df = pd.DataFrame(auc_list, index = alleles.tolist())
    print(auc_df)
    

# dataset_filename = os.path.join(data_path, "evalset_11mer_normalization.csv")
# Process_LengthFree_Prediction(dataset_filename)

def get_evaluation_by_allele():
    
    path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(path)
    alleles = dataset.allele.unique().tolist()
    print(alleles)
    for allele in alleles:
        data = ep.get_evaluation_set(allele, length=9)
        X = data.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1)
        y = data.log50k
        aw = re.sub('[*:]','_',allele) 
        result = evaluate_predictor(X, y, aw)
        # print(result)
        comp.append(result)

    