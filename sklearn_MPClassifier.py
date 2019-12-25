#sklearn_MLPClassifier_MHC

import os, sys, math
import numpy as np
import pandas as pd
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
import epitopepredict as ep

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def one_hot_encode(seq):
    '''
    Encode protein sequaence, seq, to a one-dimension array.
    Encode the peptide using dummy code and join it with zeros.
    input: [string] seq (length = n)
    output: [1x20n ndarray] e
    '''
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',') #use dummy codes to encode the peptide
    a = a.join(x) #join the peptide code with all-zero codes
    a = a.sort_index(axis=1) #sort the DataFrame based on column
    # print(a)
    e = a.to_numpy().flatten() #convert to one-dimension array
    # print(e)
    return e

nlf = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)
def nlf_encode(seq):    
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)  
    e = x.values.flatten()
    return e

blosum62 = ep.blosum62

def blosum_encode(seq):
    #encode a peptide into blosum features
    s=list(seq)
    x = pd.DataFrame([blosum62[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten() 
    print(e)   
    return e

def random_encode(p):
    return [np.random.randint(20) for i in pep]

pep='ALDFEQEMT'
#e=blosum_encode(pep)
e = one_hot_encode(pep)