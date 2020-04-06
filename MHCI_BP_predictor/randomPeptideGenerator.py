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
import os.path

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

def randomPeptideGenerator(AAfile, TranscribeTableNum, l, seqNum):
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

AAfile = os.path.join(current_path, "AAUsage.csv")
randomAASequences = randomPeptideGenerator(AAfile, 11, 8, 100)


