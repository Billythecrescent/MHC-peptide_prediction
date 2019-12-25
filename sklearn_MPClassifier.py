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

from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor

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
    e = x.to_numpy().flatten()
    return e


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

def random_encode(pep):
    return [np.random.randint(20) for i in pep]

def auc_score(true,sc,cutoff=None):

    if cutoff!=None:
        true = (true<=cutoff).astype(int)
        sc = (sc<=cutoff).astype(int)        
    fpr, tpr, thresholds = metrics.roc_curve(true, sc, pos_label=1)
    r = metrics.auc(fpr, tpr)
    #print (r)
    return  r

def test_predictor(allele, encoder, ax):

    reg = MLPRegressor(hidden_layer_sizes=(20), alpha=0.01, max_iter=500,
                        activation='relu', solver='lbfgs', random_state=2)
    df = ep.get_training_set(allele, length=9)
    print (len(df))
    X = df.peptide.apply(lambda x: pd.Series(encoder(x)),1)
    y = df.log50k
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    reg.fit(X_train, y_train)
    sc = reg.predict(X_test)
    x=pd.DataFrame(np.column_stack([y_test,sc]),columns=['test','predicted'])
    x.plot('test','predicted',kind='scatter',s=20,ax=ax)
    ax.plot((0,1), (0,1), ls="--", lw=2, c=".2")
    ax.set_xlim((0,1));  ax.set_ylim((0,1))
    ax.set_title(encoder.__name__)    
    auc = ep.auc_score(y_test,sc,cutoff=.426)
    ax.text(.1,.9,'auc=%s' %round(auc,2))
    sns.despine()


def main():
    pep='ALDFEQEMT'
    # e=blosum_encode(pep)
    # e = one_hot_encode(pep)

    sns.set_context('notebook')
    encs=[blosum_encode,nlf_encode,one_hot_encode,random_encode]
    allele='HLA-A*03:01'
    fig,axs=plt.subplots(2,2,figsize=(10,10))
    axs=axs.flat
    i=0
    for enc in encs:
        test_predictor(allele,enc,ax=axs[i])
        i+=1


main()