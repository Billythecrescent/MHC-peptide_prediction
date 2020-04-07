import os, re
import pandas as pd
import epitopepredict as ep
from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.neural_network import MLPRegressor

import MHCI_BP_predictor
import MHCI_BP_evaluator

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) #code\MHC-peptide_prediction
current_path = os.path.dirname(os.path.abspath(__file__)) #code\MHC-peptide_prediction\MHCI_BP_predictor
model_path = os.path.join(module_path,"models") #code\MHC-peptide_prediction\models
data_path = os.path.join(module_path,"data") #code\MHC-peptide_prediction\data

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
        reg = build_predictor(training_data, a, MHCI_BP_predictor.blosum_encode, 20)
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
    auc = MHCI_BP_evaluator.auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def evaluation_prediction_model(dataset_filename, length):
    auc_list = []
    df = pd.read_csv(dataset_filename)
    alleles = df.allele.unique()
    print(alleles, len(alleles))
    for allele in alleles:

        data = df.loc[df['allele'] == allele]
        X = data.peptide.apply(lambda x: pd.Series(MHCI_BP_evaluator.blosum_encode(x)),1)
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


## Build predictor ##
def build_model_controller():
    # data8mer = pd.read_csv(os.path.join(data_path, "ep_8mer_training_data.csv"))
    # data10mer = pd.read_csv(os.path.join(data_path, "ep_10mer_training_data.csv"))
    data11mer = pd.read_csv(os.path.join(data_path, "ep_11mer_training_data.csv"))
    # print(data8mer)
    build_prediction_model(data11mer, 11, 20)


# build_model_controller()

## Evaluate predictor ##
# dataset_filename = os.path.join(data_path, "evalset_8mer_normalization.csv")
# dataset_filename = os.path.join(data_path, "evalset_11mer_normalization.csv")
# evaluation_prediction_model(dataset_filename, 11)