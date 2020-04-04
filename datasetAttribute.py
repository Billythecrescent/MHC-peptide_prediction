#datasetAttribute.py

#learn the attributes of the dataset
import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep

module_path = os.path.abspath(os.path.dirname(__file__)) #path to module
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

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

def datasetDistribute(dataset, format = None, output_filename = None, IC50threshold = 500):
    '''
    Analyze the distrubution of the data frequency of alleles
        in the dataset
    'format' is the output format of the result, typically 'csv'
        Default: None  
    '''
    alleles = dataset.allele.to_numpy()
    # print(alleles)
    allele_unique, allele_counts = np.unique(alleles, return_counts=True)

    binder_counts = []
    for allele in allele_unique:
        allele_df = dataset.loc[dataset['allele'] == allele]
        allele_binder_df = allele_df.loc[allele_df['ic50'] < IC50threshold]
        binder_counts.append(len(allele_binder_df))
        # print(allele_binder_df)
        # print(len(allele_binder_df))
    
    # print(len(allele_counts), len(binder_counts))
    Combined_counts = np.array([allele_counts, binder_counts], np.int64)
    Combined_counts_2d = pd.DataFrame(Combined_counts.T, columns = ['total_counts', 'binder_counts'], index = allele_unique)
    print(Combined_counts_2d)

    if format == 'csv':
        if output_filename == None:
            Combined_counts_2d.to_csv(os.path.join(module_path, 'dataset_alletes_distribution.csv'))
        elif output_filename != None:
            Combined_counts_2d.to_csv(os.path.join(module_path, output_filename + ".csv"))
    return Combined_counts_2d



def datasetOutput(dataset, format = None, output_filename = None):
    '''
    '''
    if format == "csv":
        if output_filename == None:
            dataset.to_csv(os.path.join(module_path, "dataset_output.csv"))
        elif output_filename != None:
            dataset.to_csv(os.path.join(module_path, output_filename + ".csv"))

# train_set1 = ep.get_training_set(length=8)
# train_set2 = ep.get_training_set(length=10)
# train_set3 = ep.get_training_set(length=11)
# evalset = ep.get_evaluation_set()
# print(evalset)
# print(datasetAllele(train_set, True))
# data10mer = pd.read_csv(os.path.join(data_path, "ep_11mer_training_data.csv"))
# print(datasetDistribute(data10mer, 'csv', "Distri_ep_11mer_training_data"))

data_allmer = pd.read_csv(os.path.join(data_path, "mhci.20130222.csv"))
datasetDistribute(data_allmer, 'csv', "Distri_mhci_allmer_data")

# IEDB_mhci_dataset = pd.read_csv(os.path.join(data_path, "bdata.20130222.mhci.csv"))
# # print(IEDB_mhci_dataset)
# print(datasetDistribute(IEDB_mhci_dataset, 'csv', "distribution_bdata.20130222.mhci"))