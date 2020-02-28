#datasetAttribute.py

#learn the attributes of the dataset
import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep


def datasetAllele(dataset):
    alleles = dataset.allele.unique().tolist()
    rename_allele = []
    for allele in alleles:
        rename_allele.append(re.sub('[*]','',allele))
    return rename_allele

evalset = ep.get_evaluation_set()
# print(datasetAllele(evalset))
a = ''
for i in datasetAllele(evalset):
    a = a + i +','

print(a)
