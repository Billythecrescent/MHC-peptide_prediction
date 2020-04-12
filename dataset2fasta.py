#dataset2fasta.py

import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep

module_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

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
        print('>' + sample['allele'] + '_' + str(index), file = f)
        print(sample['peptide'] + '\n', file = f)
    f.close()

    return True


# evalset = ep.get_evaluation_set(length=9)
# filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '_' + "dataset.fasta"
# dataset2fasta(evalset, filename)
# print(evalset)

allele = "Patr-A*0101"
aw = re.sub('[*:]','_',allele)
filepath = os.path.join(data_path, "modified_mhc.20130222.csv")
df = pd.read_csv(filepath)
allele_dataset = df.loc[df['allele'] == allele]
dataset2fasta(allele_dataset, os.path.join(module_path, aw + "_" + "modified_mhc.20130222.fasta"))
