#datasetAttribute.py

#learn the attributes of the dataset
import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep


def datasetAllele(dataset, rename = False, be_substituted = '', substitute = ''):
    '''
    'be_substituted' is the standard pattern which is to find
        the char to be substituted. Default: '', referred: '[*]'.
    'substitute' is the standard pattern which is to substitute
        the char found by 'be_substituted' pattern.
    '''
    alleles = dataset.allele.unique().tolist()
    if rename == True:
        renamed_allele = []
        for allele in alleles:
            renamed_allele.append(re.sub(be_substituted, substitute, allele))
        return renamed_allele
    else:
        return alleles

def datasetDistribute(dataset, format = None):
    '''
    Analyze the distrubution of the data frequency of alleles
        in the dataset
    'format' is the output format of the result, typically 'csv'
        Default: None  
    '''
    alleles = dataset.allele.to_numpy()
    # print(alleles)
    allele_unique, allele_counts = np.unique(alleles, return_counts=True)
    allele_counts_2d = pd.DataFrame(allele_counts, columns = ['counts'], index = allele_unique)
    if format == 'csv':
        allele_counts_2d.to_csv(os.path.join('dataset_alletes_distribution.csv'))
    return allele_counts_2d


evalset = ep.get_evaluation_set()
# print(datasetAllele(evalset, True, '[*]', ''))
print(datasetDistribute(evalset, 'csv'))

