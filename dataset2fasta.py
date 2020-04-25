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


def main():
    filepath = os.path.join(data_path, "VACV_evaluation_dataset.csv")
    df = pd.read_csv(filepath)
    alleles = df.allele.unique().tolist()
    print(alleles)
    for allele in alleles:
        aw = re.sub('[*:]','_',allele)
        allele_dataset = df.loc[df['allele'] == allele]
        outputPath = os.path.join(module_path, aw + "_" + "VACV_evaluation_dataset.fasta")
        print(outputPath)
        dataset2fasta(allele_dataset, outputPath)

main()
