#./data/dataStandardization.py

import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep

module_path = os.path.dirname(os.path.abspath(__file__)) #path to module
data_path = os.path.join(os.path.abspath(os.path.dirname(module_path)),"data")

def data_8mer_normalization(filename):
    df = pd.read_csv(filename)
    alleles = df.allele.to_list()
    new_alleles = []
    for allele in alleles:
        str_groups = re.search(r'(.*\*)([0-9]{2})([0-9]{2})', allele).groups()
        new_allele = str_groups[0] + str_groups[1] + ':' + str_groups[2]
        # print(new_allele)
        new_alleles.append(new_allele) 
    new_alleles_df = pd.DataFrame(new_alleles,columns=['allele'])
    # new_df = new_alleles_df+df['peptide']+df['ic50']+df['log50k']
    new_df = pd.concat([new_alleles_df, pd.DataFrame(df, columns=['peptide', 'ic50', 'log50k'])], axis=1)
    return new_df

# ##normalize_9mer
# file = os.path.join(data_path, "evalset_9mers.csv")
# # print(file)
# # print(data_8mer_normalization(file))
# df = data_8mer_normalization(file)
# df.to_csv(os.path.join(data_path, 'evalset_9mer_normalization.csv'))

def data_9mer_normalization(filename):
    return 

def data_10mer_normalization(filename):
    return 

def data_11mer_normalization(filename):
    return 