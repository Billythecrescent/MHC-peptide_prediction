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
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
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
pseudo_path = os.path.join(current_path, "pseudo")

blosum_encode = PF.blosum50_encode

def ProcessMHCfile(species, dataset):
    alleles = [allele for allele in dataset.allele.unique().tolist() if allele[:(len(species))] == species]
    Alist = [allele for allele in alleles if allele[(len(species)+1)] == 'A']
    Blist = [allele for allele in alleles if allele[(len(species)+1)] == 'B']
    Clist = [allele for allele in alleles if allele[(len(species)+1)] == 'C']
    Elist = [allele for allele in alleles if allele[(len(species)+1)] == 'E']
    
    chosen_list = []
    deal_allele_list = []
    with open(os.path.join(mhc_path, species+'-A.fasta'), "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            #recore['id', 'name', 'discription', 'Seq']
            # print(record.id, record.name)
            for allele in Alist:
                find_index = record.description.find(allele[4:])
                if find_index != -1 and ord(record.description[find_index+7]) not in range(48,58) and allele not in deal_allele_list:
                    record.description = allele + ' | ' + str(len(record.seq))
                    chosen_list.append(record)
                    deal_allele_list.append(allele)
                    break
    with open(os.path.join(mhc_path, species+'-B.fasta'), "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            for allele in Blist:
                find_index = record.description.find(allele[4:])
                if find_index != -1 and ord(record.description[find_index+7]) not in range(48,58) and allele not in deal_allele_list:
                    record.description = allele + ' | ' + str(len(record.seq))
                    chosen_list.append(record)
                    deal_allele_list.append(allele)
                    break
    with open(os.path.join(mhc_path, species+'-C.fasta'), "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            for allele in Clist:
                find_index = record.description.find(allele[4:])
                if find_index != -1 and ord(record.description[find_index+7]) not in range(48,58) and allele not in deal_allele_list:
                    record.description = allele + ' | ' + str(len(record.seq))
                    chosen_list.append(record)
                    deal_allele_list.append(allele)
                    break
    with open(os.path.join(mhc_path, species+'-E.fasta'), "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            for allele in Elist:
                find_index = record.description.find(allele[4:])
                if find_index != -1 and ord(record.description[find_index+7]) not in range(48,58) and allele not in deal_allele_list:
                    record.description = allele + ' | ' + str(len(record.seq))
                    chosen_list.append(record)
                    deal_allele_list.append(allele)
                    break
    
    if len(deal_allele_list) != len(alleles):
        print("%d alleles have not included in your file yet." %(len(alleles)-len(deal_allele_list)))
        print([x for x in alleles if x not in deal_allele_list])
        return
    print(deal_allele_list)
    
    # print(len(chosen_list)) #89
    SeqIO.write(chosen_list, os.path.join(mhc_path, "HLA.fasta"), "fasta")

def test_ProcessMHCfile():
    allmer_data = pd.read_csv(os.path.join(data_path, "modified_mhc.20130222.csv"))
    ProcessMHCfile("HLA", allmer_data)

def loadMHCSeq(species_list = ['HLA', 'H-2', 'SLA', 'Mamu', 'Patr']):
    '''load MHC sequences from fasta file to dictionary
    species: string
        the list of name of the mhc species
        
    Return:
    -------
    MHCseqDic: dictionary
        the dictionary format of sequences, key is the allele, value is the seq
    '''
    MHCseqDic = {}
    for species in species_list:
        filepath = os.path.join(mhc_path, species+'.fasta')
        with open(filepath, "rU") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # print(record.description)
                if species == 'HLA':
                    key = re.search(r'(HLA-[ABCE]\*[0-9]{2}:[0-9]{2})', record.description).groups()[0]
                    value = record.seq
                    MHCseqDic[key] = value
                elif species == 'SLA':
                    key = re.search(r'(SLA-[0-9]\*[0-9]{2}:[0-9]{2})', record.description).groups()[0]
                    # print(key)
                    value = record.seq
                    MHCseqDic[key] = value
                elif species == 'H-2':
                    key = re.search(r'(H-2-[A-Z][a-z])', record.description).groups()[0]
                    # print(key)
                    value = record.seq
                    MHCseqDic[key] = value
                elif species == 'Mamu':
                    key = re.search(r'(Mamu-.*\*[0-9]*:[0-9]{2}?)', record.description).groups()[0]
                    # print(key)
                    value = record.seq
                    MHCseqDic[key] = value
                elif species == 'Patr':
                    key = re.search(r'(Patr-[AB]\*[0-9]{2}:[0-9]{2})', record.description).groups()[0]
                    # print(key)
                    value = record.seq
                    MHCseqDic[key] = value
    # print(MHCseqDic)
    # print(len(MHCseqDic))

    return MHCseqDic

# loadMHCSeq(['HLA', 'H-2', 'SLA', 'Mamu', 'Patr'])

def pseudoSeqGenerator(MHCseq, pseudoPosition):
    '''Generate MHC pseudo sequence for a MHCseq
    MHCseq: string or Bio.Seq
        the sequence of MHC molecules
    pseudoPosition: int[]
        the list containing the pseudo parttern (position)
    
    Return:
    ------
    pseudoSeq: string
        the pseudo sequence of a MHCseq
    '''
    pseudoSeq = ''
    if pseudoPosition[-1] > len(MHCseq):
        print("LengthException: the length of MHCseq is less than that of the MHC binding core.")
        return
    for pos in pseudoPosition:
        pseudoSeq = pseudoSeq + MHCseq[pos+1]
    # print(pseudoSeq)

    return pseudoSeq

def test_pseudoSeqGenerator():
    pseudoPositionDic = {"HLA": PC.HLA_pseudo_sequence, "SLA": PC.SLA_pseudo_sequence, "H-2": PC.H_2_pseudo_sequence, \
    "Mamu": PC.Mamu_pseudo_sequence, "NetMHC": PC.NetMHC_pseudo_sequence, "global_core": PC.globel_pseudo_sequence_core, \
        "global_general": PC.globel_pseudo_sequence_general}
    MHCseq = loadMHCSeq()
    dataset = pd.read_csv(os.path.join(data_path, "modified_mhc.20130222.csv"))
    alleles = dataset.allele.unique().tolist()

    for key in pseudoPositionDic:
        record_list = []
        for allele in alleles:
            allele_seq = pseudoSeqGenerator(MHCseq[allele], pseudoPositionDic[key])
            record = SeqRecord(Seq(allele_seq, IUPAC.protein), id=key+"_"+allele, name=allele+" pseudoSeq",
                   description="generated pseudo sequence of MHC allele")
            record_list.append(record)
        SeqIO.write(record_list, os.path.join(pseudo_path, key+"_pseudoSeqs.fasta"), "fasta")

# test_pseudoSeqGenerator()

def EncodeTo9mer(seq, blosum_encode):
    '''Transform allmer sequence to potential 9mer binding core
        as described in NetMHC4.0, different from NetMHC3.0 L-mer approximation
    seq: string
        the sequence of peptide, length of which is 8, 9, 10, 11, or other.

    Return:
    -------
    seq_list: string[]
        the list of potential binding core
    combined_list: list[int[]]
        the list of encoding nodes without the peptide itself
        encoding note: like [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            first 4 nodes represent peptide length, middle 3 represent insertion/deletion
            length, last 3 represent insertion/deletion position
    '''
    affiCoreLen = 9
    seqLen = len(seq)
    seq_list = []
    length_pep_list = []
    inserAdele_len_list = []
    inserAdele_pos_list = []
    if seqLen <= 8:
        inseLen = affiCoreLen - seqLen
        #turn 8mer to 9mer
        for i in range(seqLen+1):
            seq_list.append(blosum_encode(seq[:i] + 'X'*inseLen + seq[i:]))
            length_pep_list.append(np.array([1, 0, 0, 0]))
            inserAdele_len_encoded = [1, 1, 1] if inseLen > 6 else PF.DecToBinEncode(inseLen, 0, 6)
            inserAdele_len_list.append(np.array(inserAdele_len_encoded))
            inserAdele_pos_encoded = [1, 0, 0] if i == 0 else ([0, 0, 1] if i == seqLen else [0, 1, 0])
            inserAdele_pos_list.append(np.array(inserAdele_pos_encoded))

    elif seqLen == 9:
        seq_list.append(blosum_encode(seq))
        length_pep_list.append(np.array([0, 1, 0, 0]))
        inserAdele_len_list.append(np.array([0, 0, 0]))
        inserAdele_pos_list.append(np.array([0, 0, 0]))

    elif seqLen >= 10:
        deleLen = seqLen - affiCoreLen
        #turen 10mer to 9mer
        for i in range(affiCoreLen+1):
            seq_list.append(blosum_encode(seq[:i] + seq[(i+deleLen):]))
            if seqLen == 10:
                length_pep_list.append(np.array([0, 0, 1, 0]))
            else:
                length_pep_list.append(np.array([0, 0, 0, 1]))
            inserAdele_len_encoded = [1, 1, 1] if deleLen > 6 else PF.DecToBinEncode(deleLen, 0, 6)
            inserAdele_len_list.append(np.array(inserAdele_len_encoded))
            inserAdele_pos_encoded = [1, 0, 0] if i == 0 else ([0, 0, 1] if i == affiCoreLen else [0, 1, 0])
            inserAdele_pos_list.append(np.array(inserAdele_pos_encoded))
    
    #combine the columns to form a dataframe
    encoded_list = []
    if len(seq_list) == len(length_pep_list) == len(inserAdele_len_list) ==len(inserAdele_pos_list):
        for i in range(len(seq_list)):
            encoded = np.concatenate((seq_list[i], length_pep_list[i], inserAdele_len_list[i], inserAdele_pos_list[i]), axis=0)
            # print(encoded.shape)
            encoded_list.append(encoded)
            # encoded_list.append(length_pep_list[i] + inserAdele_len_list[i] + inserAdele_pos_list[i])
    else:
        print("the dimension of the encoding list is not consistent")
        return 

    # print(len(encoded_list))
    return encoded_list

# EncodeTo9mer("ABCDEFGH", blosum_encode)

def AllmerPanEncoder(pseudoSeq, seq, blosum_encode):
    '''Encode the peptide of allmer, with the encoding of peptide length, 
        deletion/insertion length, and deletion/insertion position
    seq: string
        the sequence of the peptide, with the length of 8, 9, 10, 11 or other
    blosum_encode: string
        the name of the blosum_encode function

    Return:
    -------
    X: numpy.array
        the encoding array of peptide, peptide feature and pseudo sequence
    '''

    encoded = EncodeTo9mer(seq, blosum_encode)
    pseudoX = blosum_encode(pseudoSeq)
    X = [np.concatenate((encoded[i], pseudoX), axis=0) for i in range(len(encoded))]
    
    X = np.array(X)
    
    return X    # the shape of X is (9, 1186) for HLApseido, is (9, 1042) for NetMHCpseudo

def test_AllmerPanEncoder():
    MHCSeqDic = loadMHCSeq()
    initialMHC = MHCSeqDic['HLA-A*01:01']
    pseudoPosition = PC.NetMHC_pseudo_sequence
    pseudoSeq = pseudoSeqGenerator(initialMHC, pseudoPosition)
    print(AllmerPanEncoder(pseudoSeq, "ABCDEFGH", blosum_encode))

# test_AllmerPanEncoder()

def AllmerPanPrepredict(pseudoSeq, seq, blosum_encode, reg, state=False):
    '''Preprediction of the peptide, to find out the binding core and encode the peptide
    pseudoSeq: string 
        the pseudo sequecen of MHC allele
    seq: string
        the sequence of the peptide, with the length of 8, 9, 10, 11 or other
    blosum_encode: string
        the name of the blosum_encode function
    reg: regression predictor
        the initialized regression predictor
    state: boolean
        the indicator of whether it is random start of exist start
        True if exist-start
        False if random-start

    Return:
    -------
    trueX: numpy.ndarray
        the only true encoding of the sequence
    '''
    X = AllmerPanEncoder(pseudoSeq, seq, blosum_encode)
    peptideX = np.split(X, [216], axis=1)[0]
    pseudoSeqX = np.split(X, [226], axis=1)[1]
    # print(peptideX.shape, pseudoSeqX.shape)
    seqX = np.concatenate((peptideX, pseudoSeqX), axis = 1)

    #predict
    if state == True:
        scores = [reg.predict(seqX[i].reshape(1,-1)) for i in range(len(seqX))]
    else:
        scores = [reg.predict(X[i].reshape(1,-1)) for i in range(len(X))]

    # print(scores)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    # print(max_score_index)
    
    trueX = X[max_score_index]
    # print(trueX, trueX.shape)
    # print(seq+"_"+"done", len(seq))
    return trueX

def test_AllmerPanPrepredict():
    MHCSeqDic = loadMHCSeq()
    initialMHC = MHCSeqDic['HLA-A*01:01']
    pseudoPosition = PC.NetMHC_pseudo_sequence
    pseudoSeq = pseudoSeqGenerator(initialMHC, pseudoPosition)
    peptide = "WEABCDEFGH"
    fname = os.path.join(os.path.join(model_path, "pan"), "BasicMHCIpan.joblib")
    if os.path.exists(fname):
        ExistReg = joblib.load(fname)
    else:
        ExistReg = None
    AllmerPanPrepredict(pseudoSeq, peptide, blosum_encode, ExistReg, True)

# test_AllmerPanPrepredict()

def RandomStartPanPredictor(dataset, hidden_node, pseudo_position, blosum_encode):
    '''Prediction of specific allele with initial random-set predictor, 
        perform cross validation
    dataset: Dataframe
        MUST has 'allele', 'peptide', 'log50k', 'length' columns
    blosum_encode: string
        the name of the blosum_encode function
    hidden_node: int
        the number of hidden layer nodes
    pseudo_position: int[]
        the pseudo position list used in the prediction
        Default: General_pseudo_position

    Return:
    -------
    reg: regression predictor
        The trained predictor
    auc_df: DataFrame
        The auc value of all prediction circles
    '''
    # if len(dataset)<200:
    #     return
    
    MHCSeqDic = loadMHCSeq()

    ##initialize the predictor
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=5000,
                        activation='relu', solver='adam', random_state=2)

    #create random X and Y as the data for regression initialization
    randomPep = PF.randomPeptideGenerator(11, 9, 1)
    initialMHC = MHCSeqDic['HLA-A*01:01']
    iniPseudoSeq = pseudoSeqGenerator(initialMHC, pseudo_position)

    iniX = AllmerPanEncoder(iniPseudoSeq, randomPep[0], blosum_encode)
    iniY = [0.1]
    reg.fit(iniX, iniY)
    
    reg.fit(iniX, iniY)

    
    y = dataset.log50k.to_numpy()

    #X is encoded below

    ##cross_validation not done
    PreCirNum = 10
    fauc = os.path.join(current_path, "MHCpan-randomStart_roc_auc.csv")
    fr = os.path.join(current_path, "MHCpan-randomStart_pearson.csv")
    avg_auc_list = []
    avg_r_list = []
    for rd in range(PreCirNum):
        print("Round %d starts" % rd)
        auc_list = []
        r_list = []
        ##encode the peptide
        X = dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, reg, False)),1).to_numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for k, (train, test) in enumerate(kf.split(X, y)):
            print("Round %d fold %d starts" %(rd, k))
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            # print(scores)
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            r = PF.pearson_score(y[test], scores)
            auc_list.append(auc)
            r_list.append(r)
        
        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        if len(avg_auc_list) > 0 and avg_auc < 0.99*avg_auc_list[-1][0]:
            break
        avg_auc_list.append(np.array([avg_auc]+auc_list))
        avg_r_list.append(np.array([avg_r]+r_list))

    avg_auc_list = np.array(avg_auc_list)
    avg_r_list = np.array(avg_r_list)

    # print(avg_auc_list)
    # print(avg_r_list)
    
    auc_df = pd.DataFrame(np.array(avg_auc_list[-1]).reshape(1,-1), columns = ['avg_AUC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    r_df = pd.DataFrame(np.array(avg_r_list[-1]).reshape(1,-1), columns = ['avg_PCC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    print(auc_df)
    print(r_df)

    # auc_df.to_csv(fauc)
    # r_df.to_csv(fr)
    
    return reg, auc_df, r_df

def test_RandomStartPanPredictor():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    # shuffled_dataset = shuffle(dataset, random_state=0)
    # allele = 'Patr-A*01:01'
    pseudoPosition = PC.HLA_pseudo_sequence
    small_dataset = dataset.loc[dataset['length'] == 12]
    shuffled_dataset = shuffle(small_dataset, random_state=0)
    # print(shuffled_dataset)
    # print(shuffled_dataset.allele.unique())
    hidden_node = 10
    RandomStartPanPredictor(shuffled_dataset, hidden_node, pseudoPosition, blosum_encode)

# test_RandomStartPanPredictor()

def Basic9merPanPrediction(dataset, hidden_node, pseudoName, pseudo_position, blosum_encode):
    MHCSeqDic = loadMHCSeq()
    y = dataset.log50k.to_numpy()

    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    X = dataset.apply(lambda x: pd.Series(blosum_encode(x.peptide+pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position))),1).to_numpy()

    reg.fit(X,y) 

    #store the predictor
    PanModelPath = os.path.join(model_path, "pan")
    fname = os.path.join(PanModelPath, "BasicMHCIpan_" + pseudoName +".joblib")
    if reg is not None:
        joblib.dump(reg, fname, protocol=2)
        print("basic MHCpan predictor is done.")
        print("Model path: %s" %fname)

def test_Basic9merPanPrediction():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    pseudoPosition = PC.NetMHC_pseudo_sequence
    dataset = dataset.loc[dataset['length'] == 9]
    shuffled_dataset = shuffle(dataset, random_state=0)
    # print(shuffled_dataset)
    # print(shuffled_dataset.allele.unique())
    # print(pseudoPosition)
    hidden_node = 20
    Basic9merPanPrediction(shuffled_dataset, hidden_node, "NetMHC", pseudoPosition, blosum_encode)

# test_Basic9merPanPrediction()

def Basic9merPanCrossValid(X, y, hidden_node, blosum_encode):
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    auc_list = []
    r_list = []
    kf = KFold(n_splits=5, shuffle=True)
    for k, (train, test) in enumerate(kf.split(X, y)):
        print("Hidden nodee:%d, fold %d starts" %(hidden_node, k))
        t0 = time()
        reg.fit(X[train], y[train])
        scores = reg.predict(X[test])
        auc = PF.auc_score(y[test], scores, cutoff=.426)
        r = PF.pearson_score(y[test], scores)
        auc_list.append(auc)
        r_list.append(r)
        t1 = time()
        print("fold %d done, run in Elapsed time %d(m)" %(k, (t1-t0)/60))
    print(auc_list)
    print(r_list)
    avg_auc = np.mean(auc_list)
    avg_r = np.mean(r_list)

    return avg_auc, avg_r

def test_Basic9merPanCrossValid():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[dataset['length'] == 9]
    pseudoPosition = PC.HLA_pseudo_sequence
    shuffled_dataset = shuffle(dataset, random_state=0)
    # print(shuffled_dataset)
    # print(shuffled_dataset.allele.unique())
    # print(pseudoPosition)
    MHCSeqDic = loadMHCSeq()
    y = dataset.log50k.to_numpy()
    X = dataset.apply(lambda x: pd.Series(blosum_encode(x.peptide+pseudoSeqGenerator(MHCSeqDic[x.allele], pseudoPosition))),1).to_numpy()

    AUClist = []
    PCClist = []
    header = pd.DataFrame(np.array(["AUC", "PCC"]).reshape(1, -1), index=["hidden node"])
    header.to_csv(os.path.join(current_path, "basicPan_crossValidation.csv"), mode='a', header=False)
    HiddenRange = range(41, 60)
    for i in HiddenRange:
        auc, r = Basic9merPanCrossValid(X, y, i, blosum_encode)
        score = pd.DataFrame(np.array([auc, r]).reshape(1, -1), columns=["AUC", "PCC"], index=[str(i)])
        score.to_csv(os.path.join(current_path, "basicPan_crossValidation.csv"), mode='a', header=False)
        AUClist.append(auc)
        PCClist.append(r)
    Scoredf = pd.DataFrame((AUClist,PCClist), columns=[i for i in HiddenRange], index = ["AUC", "PCC"])
    print(Scoredf)

# test_Basic9merPanCrossValid()

def ExistStartPanPredictor(dataset, hidden_node, pseudoName, pseudo_position, blosum_encode):
    MHCSeqDic = loadMHCSeq()
    y = dataset.log50k.to_numpy()

    fname = os.path.join(os.path.join(model_path, "pan"), "BasicMHCIpan_" + pseudoName +".joblib")
    if os.path.exists(fname):
        ExistReg = joblib.load(fname)
    else:
        ExistReg = None

    if ExistReg is None:
        print ('Locals do not have basic MHCpan model.')
        return  

    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    ##cross_validation not done
    PreCirNum = 10
    fauc = os.path.join(current_path, "MHCpan-ExistStart_roc_auc.csv")
    fr = os.path.join(current_path, "MHCpan-ExistStart_pearson.csv")
    avg_auc_list = []
    avg_r_list = []
    for rd in range(PreCirNum):
        print("Round %d starts" % rd)
        auc_list = []
        r_list = []
        ##encode the peptide
        if rd == 0:
            X = dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, ExistReg, True)),1).to_numpy()
        else:
            X = dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, reg, False)),1).to_numpy()

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for k, (train, test) in enumerate(kf.split(X, y)):
            print("Round %d fold %d starts" %(rd, k))
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            # print(scores)
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            r = PF.pearson_score(y[test], scores)
            auc_list.append(auc)
            r_list.append(r)
        
        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        if len(avg_auc_list) > 0 and avg_auc < 0.995*avg_auc_list[-1][0]:
            break
        avg_auc_list.append(np.array([avg_auc]+auc_list))
        avg_r_list.append(np.array([avg_r]+r_list))

    avg_auc_list = np.array(avg_auc_list)
    avg_r_list = np.array(avg_r_list)

    # print(avg_auc_list)
    # print(avg_r_list)
    
    auc_df = pd.DataFrame(np.array(avg_auc_list[-1]).reshape(1,-1), columns = ['avg_AUC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    r_df = pd.DataFrame(np.array(avg_r_list[-1]).reshape(1,-1), columns = ['avg_PCC']+[str(i)+"-fold" for i in range(1, 6)], index=[str(hidden_node)])
    print(auc_df)
    print(r_df)

    # auc_df.to_csv(fauc)
    # r_df.to_csv(fr)
    
    return reg, auc_df, r_df

def test_ExistStartPanPredictor():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    # shuffled_dataset = shuffle(dataset, random_state=0)
    # allele = 'Patr-A*01:01'
    # pseudoPosition = PC.HLA_pseudo_sequence
    pseudoPosition = PC.NetMHC_pseudo_sequence
    small_dataset = dataset.loc[dataset['length'] == 12]
    shuffled_dataset = shuffle(small_dataset, random_state=0)
    # print(shuffled_dataset)
    # print(shuffled_dataset.allele.unique())
    hidden_node = 10
    ExistStartPanPredictor(shuffled_dataset, hidden_node, pseudoPosition, blosum_encode)

# test_ExistStartPanPredictor()

def allmerPanPredictor(dataset, blosum_encode, hidden_node, ifRandomStart, pseudoName, pseudo_position):
    '''Choose prediction strategy according to ifRandomStart and perform cross validation
        random start uses initialized regression predictor to iterate fitting
        exist start uses existed model to prepredict the binding core of the 
            peptide and other features
    dataset: Dataframe
        MUST has 'allele', 'peptide', 'log50k', 'length' columns
    allele: string
        the name of the allele
    blosum_encode: string
        the name of the blosum_encode function
    hidden_node: int
        the number of hidden layer nodes
    ifRandomStart: Boolean
        Whether it is random start or exist start
    '''
    if ifRandomStart:
        reg, auc_df, r_df = RandomStartPanPredictor(dataset, hidden_node, pseudoName, pseudo_position, blosum_encode)
    else:
        reg, auc_df, r_df = ExistStartPanPredictor(dataset, hidden_node, pseudoName, pseudo_position, blosum_encode)

    return reg, auc_df, r_df

def MHCpanBuildPredictor(dataset, blosum_encode, hidden_node, ifRandomStart, pseudoName, pseudoPositionDic, score_filename):
    '''Build predictor according to whether it is random start or exist start
        random start uses initialized regression predictor to iterate fitting
        exist start uses existed model to prepredict the binding core of the 
            peptide and other features
    dataset: DataFrame
        the dataset of different alleles
    hidden_node: int
        the number of hidden layer nodes
    ifRandomStart: Boolean
        Whether it is random start or exist start
    pseudoName: string
        the name of the pseudoPosition list (the key in pseudoPositionDic)
        one in ['HLA', 'SLA', 'H-2', 'Mamu', 'NetMHC', 'global_core', 'global_general']
    pseudoPositionDic: dictionary
        the dictionary of pseudoPositions. key: the name of the position list; value: position list
    score_filename: string 
        the name of the output auc file (auc, PCC)
    
    Return:
    ------
    None
    '''
    StartType = 'RandomStart'
    if ifRandomStart == False:
        StartType = 'ExistStart'

    print("Prediction Mode: %s Prediction\nEncode Method: blosum50\nhidden node: %d\npseudoName: %s\noutput filename: %s\n" \
     %(StartType, hidden_node, pseudoName, score_filename))

    path = os.path.join(model_path, "allmerPan")
    pseudo_position = pseudoPositionDic[pseudoName]
    
    reg, auc_df, r_df= allmerPanPredictor(dataset, blosum_encode, hidden_node, ifRandomStart, pseudoName, pseudo_position)

    auc_df.to_csv(os.path.join(current_path, score_filename + "_" + pseudoName + "_auc.csv"), mode='a', header=False)
    r_df.to_csv(os.path.join(current_path, score_filename + "_" + pseudoName + "_PCC.csv"), mode='a', header=False)

    fname = os.path.join(os.path.join(path, StartType), pseudoName+'.joblib')
    if reg is not None:
        joblib.dump(reg, fname, protocol=2)
        print("%s fitting pseudo %s is done" %(StartType, pseudoName))

def leaveOnePrediction(dataset, hidden_node, outputFile):
    species_list = ["chimpanzee", "macaque", "mouse", "pig"]
    pseudoPositionDic = {"NetMHC": PC.NetMHC_pseudo_sequence, "global_core": PC.globel_pseudo_sequence_core, "HLA": PC.HLA_pseudo_sequence}
    for species in species_list:
        species_dataset = dataset.loc[dataset["species"] == species]
        other_dataset = dataset.loc[dataset["species"] != species]
        for key in pseudoPositionDic:
            pseudo_position = pseudoPositionDic[key]
            MHCSeqDic = loadMHCSeq()
            y_train = other_dataset.log50k.to_numpy()
            y_test = species_dataset.log50k.to_numpy()
            
            print("\nPrediction Mode: ExistStart Prediction\nEncode Method: blosum50\nhidden node: %d\nspecies: %s\npseudoName: %s\n" \
            %(hidden_node, species, key))

            fname = os.path.join(os.path.join(model_path, "pan"), "BasicMHCIpan_" + key +".joblib")
            if os.path.exists(fname):
                ExistReg = joblib.load(fname)
            else:
                ExistReg = None

            if ExistReg is None:
                print ('Locals do not have basic MHCpan model.')
                return  

            reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=5000,
                                activation='relu', solver='adam', random_state=2)
            ##cross_validation not done
            PreCirNum = 6
            auc_list = []
            r_list = []
            for rd in range(PreCirNum):
                print("Spieces: %s, pseudo: %s, round %d starts" %(species, key, rd))
                ##encode the peptide
                if rd == 0:
                    X_train = other_dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, ExistReg, True)),1).to_numpy()
                    X_test = species_dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, ExistReg, True)),1).to_numpy()
                else:
                    X_train = other_dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, reg, False)),1).to_numpy()
                    X_test = species_dataset.apply(lambda x: pd.Series(AllmerPanPrepredict(pseudoSeqGenerator(MHCSeqDic[x.allele], pseudo_position), x.peptide, blosum_encode, reg, False)),1).to_numpy()
                reg.fit(X_train, y_train)
                scores = reg.predict(X_test)
                # print(scores)
                auc = PF.auc_score(y_test, scores, cutoff=.426)
                r = PF.pearson_score(y_test, scores)
                if len(auc_list) > 0 and auc < 0.99*auc_list[-1]:
                    break
                auc_list.append(auc)
                r_list.append(r)
            
            auc_df = pd.DataFrame(np.array(auc_list[-1:]+auc_list).reshape(1,-1), columns = ['final']+[str(i+1)+"-round" for i in range(len(auc_list))], index=[species+"_"+key])
            r_df = pd.DataFrame(np.array(r_list[-1:]+r_list).reshape(1,-1), columns = ['final']+[str(i+1)+"-round" for i in range(len(r_list))], index=[species+"_"+key])
            
            print(auc_df)
            print(r_df)
            auc_df.to_csv(os.path.join(current_path, outputFile+'_auc.csv'), header=False, mode='a')
            auc_df.to_csv(os.path.join(current_path, outputFile+'_pcc.csv'), header=False, mode='a')


def test_leaveOnePrediction():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    datset = shuffle(dataset, random_state=0)
    hidden_node = 68
    outputFile = "MHCpan_LeaveOne_ExistStart"
    leaveOnePrediction(dataset, hidden_node, outputFile)

test_leaveOnePrediction()

def main():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    # pseudoPosition = PC.NetMHC_pseudo_sequence
    # shuffled_dataset = shuffle(dataset, random_state=0)
    # small_dataset = dataset.loc[dataset['length'] == 9]
    # print(dataset)
    # print(shuffled_dataset.allele.unique())
    # print(pseudoPosition)
    bestHidenNode = 68
    pseudoPositionDic = {"Mamu": PC.Mamu_pseudo_sequence, "NetMHC": PC.NetMHC_pseudo_sequence, "global_core": PC.globel_pseudo_sequence_core, \
        "global_general": PC.globel_pseudo_sequence_general}
    order = []
    # for key in pseudoPositionDic:
    #     pseudoPosition = pseudoPositionDic[key]
    #     data9mer = dataset.loc[dataset['length'] == 9]
    #     shuffled_dataset = shuffle(data9mer, random_state=0)
    #     hidden_node = 20
    #     Basic9merPanPrediction(shuffled_dataset, hidden_node, key, pseudoPosition, blosum_encode)
        
    for key in pseudoPositionDic:
        MHCpanBuildPredictor(dataset, blosum_encode, bestHidenNode, False, key, pseudoPositionDic, "MHCpan-ExistStart-AllPseudo")
        # MHCpanBuildPredictor(dataset, blosum_encode, bestHidenNode, True, key, pseudoPositionDic, "MHCpan-RandomStart-AllPseudo")
        order.append(key)
    print(order)

# main()