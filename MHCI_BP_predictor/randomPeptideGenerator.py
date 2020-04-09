'''
File: randomPeptideGenerator.py
Author: Mucun Hou
Date: Apr 6, 2020
Description: This script will generate random amino acid sequences with a
             given GC content either based on a amino acid usage probabitliy
Based on NullSeq_Functions.py by Sophia Liu
'''
from Bio.Seq import Seq
import NullSeq_Functions as NS
import argparse
import os.path, re
from math import log
import json

import pandas as pd
import numpy as np
import joblib
import MHCI_BP_evaluator

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

def randomPeptideGenerator(TranscribeTableNum, l, seqNum):
    '''Generate random amino acid sequences given a codon table
    TranscribeTableNum: int
        the codon table index according to NCBI
        default: 11
    l: int
        the length of the amino acid sequnce
    seqNum: int
        the number of generated random sequences
    
    Returns
    -------
    AASequences: string[]
        random amino acid sequence list
    '''
    
    AAfile = os.path.join(current_path, "AAUsage.csv")
    N = TranscribeTableNum
    if AAfile is not None:
        if AAfile.split('.')[-1] == 'csv':
            AAUsage = NS.df_to_dict(NS.AAUsage_from_csv(AAfile), N)
            length = l
            AASequence = None
            operatingmode = 'AA Usage Frequency'
        else:
            AASequence = NS.parse_fastafile(AAfile)
            AAUsage = NS.get_AA_Freq(AASequence, N, nucleotide=False)
            operatingmode = 'Existing Sequence - AA'
            if l == None:
                length = len(AASequence)-1
            else:
                length = l
            if ES:
                pass
            else:
                AASequence = None
    AASequences = []
    for i in range(seqNum):
        AASequences.append(NS.get_Random_AA_Seq(AAUsage, length))
    # print(AASequences)
    return AASequences

# randomAASequences = randomPeptideGenerator(11, 8, 100)

def find_model(allele, length):
    if length != 9:
        fname = os.path.join(os.path.join(model_path, "Non9mer"), allele + "-" + str(length) +'.joblib')
    elif length == 9:
        fname = os.path.join(model_path, allele+'.joblib')
    print(fname)
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return

def PrePredict(peptide_df, length):
    X = peptide_df.apply(lambda x: pd.Series(MHCI_BP_evaluator.blosum_encode(x)),1)
    if length == 8:
        allele = "H-2-Kb"
    elif length == 9 or length == 10:
        allele = "HLA-A*02:01"
    elif length == 11:
        allele = "Mamu-A*01:01"
    aw = re.sub('[*:]','_',allele) 
    reg = find_model(aw, length)
    if reg is None:
        print ('Locals do not have model for %s of this %s.' %(aw, length))
        scores = float('nan')*len(randomSeqs_df.peptide)
    else:
        scores = reg.predict(X)
    return scores

def AddRandomDataToDataset(DatasetFile, LengthList = [8, 9, 10, 11], SeqNum = 100, Log50Kthreshold = .361, output_filename = None):
    '''Add artificial random peptide affinity data to original dataset
    DatasetFile: string
        the file path of the csv format afinity dataset
    LengthList: int[]
        the list of the lengths to be added of each allele
    SeqNum: int
        the number of random sequences added to each length of each allele
    '''
    Dataset = pd.read_csv(DatasetFile)
    alleles = Dataset.allele.unique().tolist()
    # print(alleles, len(alleles))
    randompeptidesOnLength = {}
    affinityOnLength = {}

    if os.path.exists(os.path.join(current_path, "PeptideAffinityFile.txt")):
        store_flag = 0
        random_flag = 0
    else:
        store_flag = 1
        random_flag = 1

    if random_flag:
        for length in LengthList:
            randomSeqs = randomPeptideGenerator(11, length, SeqNum*len(alleles))
            randompeptidesOnLength[length] = randomSeqs
            randomSeqs_df = pd.DataFrame(randomSeqs, columns = ['peptide'])
            peptide_df = randomSeqs_df.peptide
            scores = PrePredict(peptide_df, length)
            affinityOnLength[length] = scores.tolist()
        
        ##use threshould to filter those which do not match the criteria (>5000nM, 0.214) and log the number and each index
        NotMatchNum = {8:0, 9:0, 10:0, 11:0}
        # NotmatchIndex = {8:[], 9:[], 10:[], 11:[]}
        PeptideAffinity = {length: list(zip(randompeptidesOnLength[length], affinityOnLength[length])) for length in range(8, 12)}
        for length in LengthList: #list copy
            pepAffs = PeptideAffinity[length]
            for pepAff in pepAffs[:]:
                if pepAff[-1] >= Log50Kthreshold:
                    # NotmatchIndex[length].append(i)
                    PeptideAffinity[length].remove(pepAff)
                    NotMatchNum[length] = NotMatchNum[length] + 1
            # for index in NotmatchIndex[length]:
            #     randompeptidesOnLength[length].pop(index)
            #     affinityOnLength[length].pop(index)
        # print(NotmatchIndex)
        # print(NotMatchNum)
        for length in LengthList:
            while NotMatchNum[length] != 0:
                newNotMatchNum = 0
                randomSeqs = randomPeptideGenerator(11, length, NotMatchNum[length])
                randomSeqs_df = pd.DataFrame(randomSeqs, columns = ['peptide'])
                peptide_df = randomSeqs_df.peptide
                scores = PrePredict(peptide_df, length).tolist()
                SeqScores = list(zip(randomSeqs, scores))
                for SeqScore in SeqScores[:]:
                    if SeqScore[-1] >= Log50Kthreshold:
                        newNotMatchNum = newNotMatchNum + 1
                        SeqScores.remove(SeqScore)
                # add the newly generated matched random peptides and affinity to the list 
                PeptideAffinity[length] = PeptideAffinity[length] + SeqScores
                #update notMatch number
                NotMatchNum[length] = newNotMatchNum
        # Test = pd.DataFrame(PeptideAffinity[8], columns = ['peptide', 'log50k'])
        # Test.to_csv(os.path.join(current_path, "test.csv"))

    else:
        # randompeptidesFile = open(os.path.join(current_path, "RandomPeptideFile.txt"), "r", encoding='UTF-8')
        # affinityFile = open(os.path.join(current_path, "affinityFile.txt"), "r", encoding='UTF-8')
        # randompeptidesOnLength = json.loads(randompeptidesFile.read())
        # affinityOnLength = json.loads(affinityFile.read())
        PeptideAffinityFile = open(os.path.join(current_path, "PeptideAffinityFile.txt"), "r", encoding='UTF-8')
        PeptideAffinity = json.loads(PeptideAffinityFile.read())
    
    ##chang non-int-type keys to int-type keys
    #Owing to json load bugs, int-type keys have been transformed to str-type
    keys = list(PeptideAffinity.keys())
    for key in keys:
        if type(key) != int:
            PeptideAffinity[int(key)] = PeptideAffinity.pop(key)
        else:
            continue

    keys = list(PeptideAffinity.keys())
    for key in keys:
        if type(key) != int:
            PeptideAffinity[int(key)] = PeptideAffinity.pop(key)
        else:
            continue
    
    # print(list(PeptideAffinity.keys()))

    if store_flag:
        # randompeptidesOnLength = json.dumps(randompeptidesOnLength)
        # affinityOnLength = json.dumps(affinityOnLength)
        # randompeptidesFile = open(os.path.join(current_path, "RandomPeptideFile.txt"), "w", encoding='UTF-8')
        # affinityFile = open(os.path.join(current_path, "affinityFile.txt"), "w", encoding='UTF-8')
        # randompeptidesFile.write(randompeptidesOnLength)
        # affinityFile.write(affinityOnLength)
        # randompeptidesFile.close()
        # affinityFile.close()
        PeptideAffinity = json.dumps(PeptideAffinity)
        PeptideAffinityFile = open(os.path.join(current_path, "PeptideAffinityFile.txt"), "w", encoding='UTF-8')
        PeptideAffinityFile.write(PeptideAffinity)
        PeptideAffinityFile.close()
    
    
    #generate allele column in Added RandomSeq Dataframe
    #colum order priority: allele, length
    allele_column = []
    #peptideAffinity_column includes the peptide and cooresponding affinity (ic50)
    peptideAffinity_column = []
    for allele in alleles:
        allele_column = allele_column + [allele]*SeqNum*len(LengthList)
        allele_index = alleles.index(allele)
        allele_peptide_affinity = []
        for length in LengthList:
            allele_peptide_affinity = allele_peptide_affinity + PeptideAffinity[length][(allele_index*SeqNum):((allele_index+1)*SeqNum)]
        peptideAffinity_column = peptideAffinity_column + allele_peptide_affinity
    # print(len(peptide_column))

    ##create "artificial" tag
    tag_column = ["artificial"]*len(LengthList)*SeqNum*len(alleles)
    species_column = ['random']*len(LengthList)*SeqNum*len(alleles)

    length_column = []
    for length in LengthList:
        length_column = length_column + [str(length)]*SeqNum
    length_column = length_column * len(alleles)

    ##Combine column and build dataframe
    DataText = np.array([species_column, allele_column, length_column, tag_column]).T
    RandomDataFrame = pd.DataFrame(np.hstack((DataText, np.array(peptideAffinity_column))), columns = ['species','allele','peptide_length','source','peptide','log50k'])
    # print(RandomDataFrame)
    # RandomDataFrame.to_csv(os.path.join(current_path, "test_df.csv"))

    ##Process natural dataset and random dataset
    NaturalDataFrame = pd.DataFrame(Dataset, columns = ['species','allele','peptide_length','source','peptide'])
    Data_ic50 = Dataset.ic50.tolist()
    Data_log50k = []
    for datum in Data_ic50:
        datum_log50k =  1 - log(datum)/log(50000)
        Data_log50k.append(datum_log50k)
    Data_log50k = pd.DataFrame(np.array(Data_log50k).T, columns = ['log50k'])
    NaturalDataFrame = pd.concat([NaturalDataFrame, Data_log50k], axis=1)

    CombinedDatframe = pd.concat([NaturalDataFrame, RandomDataFrame], axis=0)
    print(CombinedDatframe)

    if output_filename != None:
        output_path = os.path.join(current_path, output_filename + ".csv")
        CombinedDatframe.to_csv(output_path)

aller_mhci = os.path.join(data_path, "mhci.20130222.csv")
AddRandomDataToDataset(aller_mhci, output_filename = "modified_mhc.20130222")


