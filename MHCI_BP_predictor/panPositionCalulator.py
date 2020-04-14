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

def panPositionCalculator(filepath):
    '''Read the position list from csv file and output pesudo position.
    filepath: string
        The filn path of the position list
    
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
        if occurance[key] > 1:
            psedo_pos.append(key)
    print("The psedo positions and the length in below:")
    psedo_pos = sorted([ int(i) for i in psedo_pos])
    print(psedo_pos, len(psedo_pos))

    return psedo_pos


filepath = os.path.join(current_path, "allele_psedoSeq_positions.csv")
panPositionCalculator(filepath)