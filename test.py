import os, sys, math
import numpy as np
import pandas as pd
import epitopepredict as ep
from epitopepredict import base, sequtils, analysis, plotting
from epitopepredict import peptutils
import joblib
import os

def auc_score(true,sc,cutoff=None):
    '''
    Calculate the auc score of soc curve
    '''
    if cutoff!=None:
        true = (true<=cutoff).astype(int)
        sc = (sc<=cutoff).astype(int)
    # print(true, sc)
    
    r = metrics.roc_auc_score(true, sc) 
    # #Or use the following code for alternative
    # fpr, tpr, thresholds = metrics.roc_curve(true, sc, pos_label=1)
    # r = metrics.auc(fpr, tpr)
    
    return  r

def evaluate_predictor(P, allele):

    data = ep.get_evaluation_set(allele, length=9)
    #print (len(data))
    P.predict_peptides(list(data.peptide), alleles=allele, cpus=4)
    x = P.get_scores(allele)
    x = data.merge(x,on='peptide')
    auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc, data

preds = [ep.get_predictor('basicmhc1'),
         ep.get_predictor('netmhcpan',scoring='affinity'),
         ep.get_predictor('mhcflurry')]
comp=[]
evalset = ep.get_evaluation_set(length=9)
test_alleles = evalset.allele.unique()

for P in preds[:1]:    
    m=[]
    for a in test_alleles[:1]:        
        try:
            auc,df = evaluate_predictor(P, a)
            m.append((a,auc,len(df)))            
        except Exception as e:
            print (a,e)
            pass
    m=pd.DataFrame(m,columns=['allele','score','size'])
    m['name'] = P.name
    comp.append(m)

print(comp)