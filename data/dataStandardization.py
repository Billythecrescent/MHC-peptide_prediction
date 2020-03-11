#./data/dataStandardization.py

import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep

def data_8mer_normalization(filename):
    df = pd.read_csv(filename)
    alleles = df.allele.to_list()
    new_alleles = []
    for allele in alleles:
        str_groups = re.search(r'(.*\*)([0-9]{2})([0-9]{2})', allele).groups()
        new_allele = str_groups[1] + str_groups[2] + ':' + str_groups[3]
        print(new_allele)
        new_alleles.append(new_allele) 
    new_alleles_df = pd.DataFrame(new_alleles,columns=['allele'])
    new_df = new_alleles_df+df['peptide']+df['ic50']+df['log50k']
    return new_df

file = "evalset_9mers.csv"
print(data_8mer_normalization(file))

def data_9mer_normalization(filename):
    return 

def data_10mer_normalization(filename):
    return 

def data_11mer_normalization(filename):
    return 