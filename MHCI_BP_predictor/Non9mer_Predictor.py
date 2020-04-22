import os, sys, math, re
import numpy as np
import pandas as pd
from time import time
import epitopepredict as ep

from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import PredictionFunction as PF

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

blosum_encode = PF.blosum_encode

def get_allele_names(data):
    a = data.allele.value_counts()
    a =a[a>=1]
    return list(a.index)

def find_model(allele, length):
    fname = os.path.join(os.path.join(model_path, "Non9mer"), allele + "-" + str(length) +'.joblib')
    if os.path.exists(fname):
        reg = joblib.load(fname)
        return reg
    else:
        return

def build_predictor(training_data, allele, encoder, hidden_node):

    data = training_data[training_data['allele'] == allele]
    if len(data) < 1:
        return

    # #write training dataframe to csv file
    # aw = re.sub('[*:]','_',allele) 
    # data.to_csv(os.path.join('alletes',aw+'_data.csv'))
    
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=500,
                        activation='relu', solver='lbfgs', random_state=2)    
    X = data.peptide.apply(lambda x: pd.Series(encoder(x)),1) #Find bug: encoding result has NaN
    y = data.log50k

    ## ---- TEST ---- ##
    # print(X)
    # print (allele, len(X))
    
    reg.fit(X,y)       
    return reg

def build_prediction_model(training_data, length, hidden_node):
    al = get_allele_names(training_data)
    print(al)
    path = os.path.join(model_path, "Non9mer")
    for a in al:
        aw = re.sub('[*:]','_',a) 
        fname = os.path.join(path, aw + "-" + str(length) +'.joblib')
        print(aw + "-" + str(length) + ".joblib is done.")
        reg = build_predictor(training_data, a, blosum_encode, 20)
        if reg is not None:
            joblib.dump(reg, fname, protocol=2)

def evaluate_predictor(X, y, allele, length):

    #print (len(data))
    # print(list(data.peptide), allele)
    reg = find_model(allele, length)
    if reg is None:
        print ('Locals do not have model for this allele.')
        return float('nan')
    scores = reg.predict(X)

    #Generate auc value
    auc = PF.auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def evaluation_prediction_model(dataset_filename, length):
    auc_list = []
    df = pd.read_csv(dataset_filename)
    alleles = df.allele.unique()
    print(alleles, len(alleles))
    for allele in alleles:

        data = df.loc[df['allele'] == allele]
        X = data.peptide.apply(lambda x: pd.Series(PF.blosum_encode(x)),1)
        y = data.log50k
        aw = re.sub('[*:]','_',allele) 
        result = evaluate_predictor(X, y, aw, length)
        auc_list.append(result)

    print(auc_list, len(auc_list))
    # print(alleles.tolist())
    #Write Result
    auc_df = pd.DataFrame(auc_list, index = alleles.tolist())
    auc_df.columns = ['auc']
    print(auc_df)

def Non9merCrossValid(X, y, hidden_node):
    reg = MLPRegressor(hidden_layer_sizes=(hidden_node), alpha=0.01, max_iter=1000,
                        activation='relu', solver='adam', random_state=2)
    
    avg_auc_list = []
    avg_pcc_list = []
    for i in range(10):
        auc_list = []
        r_list = []
        kf = KFold(n_splits=5, shuffle=True)
        for k, (train, test) in enumerate(kf.split(X, y)):
            t0 = time()
            reg.fit(X[train], y[train])
            scores = reg.predict(X[test])
            auc = PF.auc_score(y[test], scores, cutoff=.426)
            r = PF.pearson_score(y[test], scores)
            auc_list.append(auc)
            r_list.append(r)
            t1 = time()
        # print(auc_list)
        # print(r_list)
        avg_auc = np.mean(auc_list)
        avg_r = np.mean(r_list)
        avg_auc_list.append(avg_auc)
        avg_pcc_list.append(avg_r)

    return np.mean(avg_auc_list), np.mean(avg_pcc_list)

def test_Non9merCrossValid():
    file_path = os.path.join(data_path, "modified_mhc.20130222.csv")
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[dataset['length'] != 9]
    alleles = dataset.allele.unique().tolist()
    # print(dataset)
    HiddenRange = range(5,21)
    header = pd.DataFrame(np.array(HiddenRange).reshape(1, -1), index=["hidden node"])
    header.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_auc.csv"), mode='a', header=False)
    header.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_pcc.csv"), mode='a', header=False)
    for length in [8, 10, 11]:
        length_dataset = dataset.loc[dataset['length'] == length]
        t0 = time()
        for allele in alleles:
            allele_dataset = length_dataset.loc[length_dataset['allele'] == allele]
            X = allele_dataset.peptide.apply(lambda x: pd.Series(blosum_encode(x)),1).to_numpy()
            y = allele_dataset.log50k.to_numpy()
            auc_list = []
            pcc_list = []
            for i in HiddenRange:
                auc, r = Non9merCrossValid(X, y, i)
                auc_list.append(auc)
                pcc_list.append(r)
                # score = pd.DataFrame(np.array([auc, r]).reshape(1, -1), columns=["AUC", "PCC"], index=[str(i)])
                # score.to_csv(os.path.join(current_path, "basicPan_crossValidation.csv"), mode='a', header=False)
            auc_df = pd.DataFrame(np.array(auc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele+"_"+str(length)+"mer"])
            pcc_df = pd.DataFrame(np.array(pcc_list).reshape(1,-1), columns=[i for i in HiddenRange], index = [allele+"_"+str(length)+"mer"])
            print(auc_df)
            print(pcc_df)
            auc_df.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_auc.csv"), mode='a', header=False)
            pcc_df.to_csv(os.path.join(current_path, "basicMHC_Non9mer_crossValidation_pcc.csv"), mode='a', header=False)
            print("allele %s on length %d is done" %(allele, length))
        t1 = time()
        print("length %d is done, run in Elapsed time %d(m)" %(length, (t1-t0)/60))


test_Non9merCrossValid()

## Build predictor ##
def build_model_controller():
    # data8mer = pd.read_csv(os.path.join(data_path, "ep_8mer_training_data.csv"))
    # data10mer = pd.read_csv(os.path.join(data_path, "ep_10mer_training_data.csv"))
    data11mer = pd.read_csv(os.path.join(data_path, "ep_11mer_training_data.csv"))
    # print(data8mer)
    build_prediction_model(data11mer, 11, 20)


# build_model_controller()

