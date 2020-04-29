#datasetAttribute.py

#learn the attributes of the dataset
import os, sys, math, time, re
import numpy as np
import pandas as pd
import epitopepredict as ep

module_path = os.path.abspath(os.path.dirname(__file__)) #path to module
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data


def datasetAllele(dataset, rename = False, be_substituted = '', substitute = '', output_filename = None):
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
        final_alleles = pd.DataFrame(renamed_allele, columns = ['allele'])
    else:
        final_alleles =  pd.DataFrame(alleles, columns = ['allele'])

    if output_filename != None:
        final_alleles.to_csv(os.path.join(module_path, output_filename + '.csv'))
    return final_alleles

def datasetDistribute(dataset, affinity = "ic50", format = None, output_filename = None, threshold = 500):
    '''Analyze the distrubution of the data frequency of alleles
        in the dataset
    dataset: Dataframe
        the dataset to be analyzed
    affinity: string
        the affinity format
        whether 'ic50' or 'log50k'
    format: string
        the output format of the result, typically 'csv'
        Default: None
    output_filename: string
        the ouput file name without path indicator and suffix
    threshold: double
        the threshold when needs to output the number of binders
        if $affinity == 'ic50' (representing ic50 threshold)
        <= 1 if $affinity == 'log50k' (representing log50k threshold)

    '''
    alleles = dataset.allele.to_numpy()
    # print(alleles)
    allele_unique, allele_counts = np.unique(alleles, return_counts=True)

    if affinity not in ('ic50', 'log50k'):
        print("the affinity column in dataset is not set correct name.")
        return
    if affinity == 'log50k' and threshold > 1:
        print("the value of %s threshold is not valid" %affinity)
        return

    binder_counts = []
    for allele in allele_unique:
        allele_df = dataset.loc[dataset['allele'] == allele]
        if affinity == 'ic50':
            allele_binder_df = allele_df.loc[allele_df[affinity] < threshold]
        elif affinity == 'log50k':
            allele_binder_df = allele_df.loc[allele_df[affinity] > threshold]
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


def datasetLengthDistribution(dataset, outputPath):
    lengths = dataset.length
    length_unique, length_counts = np.unique(lengths, return_counts=True)
    # print(length_counts)
    LengthDistr = pd.DataFrame(length_counts, columns=["counts"], index=length_unique)
    print(LengthDistr)
    LengthDistr.to_csv(outputPath)

def test_datasetLengthDistribution():
    mhcii_dataset = pd.read_csv(os.path.join(data_path, "mhcii-DRB1-dataset.csv"))
    dataset = mhcii_dataset.loc[mhcii_dataset["allele"] == "HLA-DRB1*0101"]
    datasetLengthDistribution(dataset, os.path.join(module_path, "Distr-mhcii-DRB1.csv"))

# test_datasetLengthDistribution()

# train_set1 = ep.get_training_set(length=8)
# train_set2 = ep.get_training_set(length=10)
# train_set3 = ep.get_training_set(length=11)
# evalset = ep.get_evaluation_set()
# print(evalset)
# print(datasetAllele(train_set, True))
# data10mer = pd.read_csv(os.path.join(data_path, "ep_11mer_training_data.csv"))
# print(datasetDistribute(data10mer, 'csv', "Distri_ep_11mer_training_data"))

# data_allmer = pd.read_csv(os.path.join(data_path, "modified_mhc.20130222.csv"))
# datasetAllele(data_allmer, output_filename = "mhci_allmer")
# dataset = data_allmer.loc[data_allmer['length'] != 9]
# datasetDistribute(dataset, 'log50k', 'csv', "Distri_mhci_non9mer_data", 0.426)

data = pd.read_csv(os.path.join(data_path, "mhci_tumor_testData.csv"))
alleles = data.allele.to_numpy()
allele_unique, allele_counts = np.unique(alleles, return_counts=True)
distri = pd.DataFrame(allele_counts, index=allele_unique)
print(distri)