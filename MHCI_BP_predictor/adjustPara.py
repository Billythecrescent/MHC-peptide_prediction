#adjustPara.py

import MHCI_BP_predictor
import MHCI_BP_evaluator
import pandas as pd
import epitopepredict as ep

def evaluate_predictor(allele, evalset, reg):
    data = evalset[evalset['allele'] == allele]
    X = data.peptide.apply(lambda x: pd.Series(MHCI_BP_predictor.blosum_encode(x)),1)
    y = data.log50k

    if reg is None:
        print ('model does not exist.')
        return
    scores = reg.predict(X)
    #Generate auc value
    auc = MHCI_BP_predictor.auc_score(y,scores,cutoff=.426) # auc = ep.auc_score(y_test,sc,cutoff=.426)
    # auc = auc_score(x.ic50,x.score,cutoff=500)
    return auc

def test_parameter():
    hidden_nodes = [i for i in range(20, 61)]
    training_data = ep.get_training_set(length=9)
    evalset = ep.get_evaluation_set(length=9) #type: DataFrame
    al = evalset.allele.unique().tolist()
    df = pd.DataFrame(index = al)
    # print(hidden_nodes)
    for hn in hidden_nodes:
        comp=[]
        # al = MHCI_BP_predictor.get_allele_names(evalset)
        
        for a in al:
            reg = MHCI_BP_predictor.build_predictor(training_data, a, MHCI_BP_predictor.blosum_encode, hn)
            result = evaluate_predictor(a, evalset, reg)
            # print(result)
            comp.append(result)
        # print(comp)
        # print(al)
        
        #Write Result
        comp_df = pd.DataFrame(comp, index = al)
        # print(comp_df)
        df[str(hn)] = comp_df
        print("hidden node: %d is done" %hn)

    print(df)
    df.to_csv("result.csv")

# test_parameter()