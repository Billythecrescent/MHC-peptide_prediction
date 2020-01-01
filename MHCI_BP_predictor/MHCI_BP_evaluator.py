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

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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

def main():
        
    preds = [ep.get_predictor('basicmhc1'),
            ep.get_predictor('netmhcpan',scoring='affinity'),
            ep.get_predictor('mhcflurry')]
    comp=[]
    evalset = ep.get_evaluation_set(length=9) #type: DataFrame

    #write training dataframe to csv file
    evalset.to_csv(os.path.join('evaluate_data.csv'))
    
    test_alleles = evalset.allele.unique() #numpy.ndarray 'str'
    
    # for P in preds:    
    #     m=[]
    #     for a in test_alleles[:1]:    
    #         auc,df = evaluate_predictor(P, a)  
    #         m.append((a,auc,len(df)))    
    #         try:
    #             auc,df = evaluate_predictor(P, a)
    #             m.append((a,auc,len(df)))            
    #         except Exception as e:
    #             print (a,e)
    #             pass
    #     m=pd.DataFrame(m,columns=['allele','score','size'])
    #     m['name'] = P.name
    #     comp.append(m)
    # print(comp)
    # #display evaluation
    # c=pd.concat(comp)
    # x=pd.pivot_table(c,index=['allele','size'],columns='name',values='score')#.reset_index()

    # ax=sns.boxplot(data=c,y='score',x='name')#,hue='allele')
    # g=sns.catplot(data=c,y='score',x='allele',hue='name',
    #             kind='bar',aspect=3,height=5,legend=False)
    # plt.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.setp(g.ax.get_xticklabels(), rotation=90)
    # plt.tight_layout()
    # plt.savefig('benchmarks.png')
    # x.to_csv('benchmarks.csv')

main()