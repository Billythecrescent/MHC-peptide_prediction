'''
File: MHCpanPrediction.py
Author: Mucun Hou
Date: Apr 14, 2020
Description: This script is to predict MHC-I binding peptide based on a pan-spesific
    method, from NetMHCpan, but use different pseudo sequence, different dataset and 
    aim at different alleles.
'''

import os, re
import pandas as pd
import numpy as np
from Bio import SeqIO
from time import time
# from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import *
import joblib
import epitopepredict as ep

import PredictionFunction as PF
import panPositionCalulator as PC

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
mhc_path = os.path.join(current_path, "MHC_proteins")

blosum_encode = PF.blosum_encode


def ProcessMHCfile(species, dataset):
    alleles = [allele for allele in dataset.allele.unique().tolist() if allele[:(len(species))] == species]
    Alist = [allele for allele in alleles if allele[(len(species)+1)] == 'A']
    Blist = [allele for allele in alleles if allele[(len(species)+1)] == 'B']
    Clist = [allele for allele in alleles if allele[(len(species)+1)] == 'C']
    Elist = [allele for allele in alleles if allele[(len(species)+1)] == 'E']
    
    with open(os.path.join(mhc_path, species+'-A.fasta'), "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            #recore['id', 'name', 'discription', 'Seq']
            # print(record.description)   #HLA:HLA24129 A*80:06 365 bp

            # 使用正则表达式匹配description中的A*80:01:01:01，从而获得序列allele
            if re.search(r'([A-Z])([0-9]{2})([0-9]{2})', record.description) != None:
                str_groups = re.search(r'([A-Z])([0-9]{2})([0-9]{2})', record.description).groups()
                new_allele = "HLA-" + str_groups[0] + "*" + str_groups[1] + ':' + str_groups[2]
                new_alleles.append(new_allele)

            # 判断该allele是否在list中（注意:01:01:01的冗余）

            # 截留对应allele的唯一序列，并储存该record


allmer_data = pd.read_csv(os.path.join(data_path, "modified_mhc.20130222.csv"))
ProcessMHCfile("HLA", allmer_data)

