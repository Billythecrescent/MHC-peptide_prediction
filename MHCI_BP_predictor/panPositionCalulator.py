'''
File: panPositionCalulator.py
Author: Mucun Hou
Date: Apr 14, 2020
Description: This script is to calculate the number of position index occurance
    of which the MHC-I molecule has interaction (within 4A) between binder.
    THIS script is used to find the psedo sequnce of MHC in pan-specific method.
'''

import os
import numpy as np
import pandas as pd

##--- File Paths ---##
module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data
pseudo_path = os.path.join(current_path, "pseudo") #code\MHC-peptide_prediction\MHCI_BP_predictor\pseudo


def panPositionCalculator(filepath, threshold):
    '''Read the position list from csv file and output pesudo position.
    filepath: string
        The filn path of the position list
    threshold: int
        The number threshold for filter
        Only occurance more than the threshold could appear in the pseudo sequcen.
    
    Return:
    ------
    sorted_psedo_pos: list
        pseodp position chosen for MHCpan predictor

    '''
    positionList = pd.read_csv(filepath).to_numpy().flatten()
    print(positionList)
    occurance = {}
    for position in positionList:
        # print(position, type(position)) # numpy.float64
        if np.isnan(position) == False:
            if position not in occurance:
                occurance[position] = 1
            else:
                occurance[position] = occurance[position] + 1
    number = len(occurance)
    print("The number of total positions are %d." %number)
    print("The raw occurance of positions is below:")
    print(occurance)
    psedo_pos = []
    for key in occurance:
        if occurance[key] > threshold:
            psedo_pos.append(key)
    print("The psedo positions and the length in below:")
    psedo_pos = sorted([ int(i) for i in psedo_pos])
    print(psedo_pos, len(psedo_pos))

    return psedo_pos

def loadPositionFile():
    HLA_filepath = os.path.join(pseudo_path, "HLA_psedoSeq_positions.csv")
    SLA_filepath = os.path.join(pseudo_path, "SLA_psedoSeq_positions.csv")
    SLA_HLA_filepath = os.path.join(pseudo_path, "SLA-HLA_psedoSeq_positions.csv")
    H_2_filepath = os.path.join(pseudo_path, "H-2_psedoSeq_positions.csv")
    H_2_HLA_filepath = os.path.join(pseudo_path, "H-2-HLA_psedoSeq_positions.csv")
    Mamu_filepath = os.path.join(pseudo_path, "Mamu_psedoSeq_positions.csv")
    Mamu_HLA_filepath = os.path.join(pseudo_path, "Mamu-HLA_psedoSeq_positions.csv")
    panPositionCalculator(Mamu_HLA_filepath, 1)


#pseudo sequence length = 40 
HLA_pseudo_sequence = [5, 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, \
     95, 97, 99, 114, 116, 118, 123, 124, 143, 146, 147, 150, 152, 155, 156, 158, 159, 163, 167, 170, 171]

#length = 45
SLA_pseudo_sequence = [6, 7, 8, 9, 45, 46, 60, 63, 64, 66, 67, 68, 70, 71, 73, 74, 77, 78, 80, 81, 82, 84, 85, 95, 99, 100, 116, \
     117, 123, 143, 144, 146, 147, 148, 150, 152, 153, 155, 156, 157, 159, 160, 163, 167, 168, 171, 172]

#length = 37
H_2_pseudo_sequence = [5, 7, 9, 22, 24, 45, 59, 62, 63, 66, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, \
     114, 116, 123, 142, 143, 146, 147, 150, 152, 155, 156, 159, 163, 167, 171]

#length = 37 (threshold = 0)
Mamu_pseudo_sequencedd = [5, 7, 9, 24, 36, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, \
     99, 114, 116, 123, 143, 146, 147, 150, 152, 155, 156, 159, 163, 167, 171]

#length = 55
SLA_HLA_pseudo_sequence = [5, 6, 7, 8, 9, 24, 45, 46, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73, 74, 76, 77, 78, 80, 81, 82, \
     84, 85, 95, 97, 99, 100, 114, 116, 123, 143, 144, 146, 147, 148, 150, 152, 153, 155, 156, 157, 159, 160, 163, 167, 168, 170, 171, 172]

#length = 40
H_2_HLA_pseudo_sequence = [5, 7, 9, 22, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116, 123, \
     124, 142, 143, 146, 147, 150, 152, 155, 156, 159, 163, 167, 170, 171]

#length = 38
Mamu_HLA_pseudo_sequence = [5, 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, \
     114, 116, 123, 124, 143, 146, 147, 150, 152, 155, 156, 159, 163, 167, 170, 171]

# print(len(pseudo_sequence))