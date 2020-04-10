#dataset2fasta.py

import os, sys, math, time
import numpy as np
import pandas as pd
import epitopepredict as ep

def dataset2fasta(dataset, filePath):
    '''Convert MHC affinity DataFrame to fasta file
    dataset: DataFrame
        MHC affinity data, must contain 'allele' and 'peptide' columns
    filename: string
        The output file path
    
    Return:
    ------
    True
    '''
    f = open(filePath,'w')
    # peptides = dataset.peptide
    # alleles = dataset.allele
    # datapair = dict(zip(alleles,peptides)) 
    # print(datapair)
    for index, sample in dataset.iterrows():
        # print(sample['allele'])
        # print(sample['peptide'])
        print('>' + sample['allele'], file = f)
        print(sample['peptide'] + '\n', file = f)
    f.close()

    return True


evalset = ep.get_evaluation_set(length=9)
filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '_' + "dataset.fasta"
dataset2fasta(evalset, filename)
# print(evalset)