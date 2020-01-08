#test_for_epitope.py

import epitopepredict as ep
from epitopepredict import base, sequtils, analysis, plotting
from epitopepredict import peptutils
import joblib
import os

module_path = os.path.dirname(os.path.abspath(__file__)) #path to module
model_path = os.path.join(module_path,"models")

#get list of predictors
# print (base.predictors)
    # ['tepitope', 'netmhciipan', 'iedbmhc1', 'iedbmhc2', 'mhcflurry', 'mhcnuggets', 'iedbbcell']
p = base.get_predictor('basicmhc1')

alleles = ["HLA-A_01_01", "HLA-A_02_01"]
seqs = peptutils.create_random_sequences(5000)
reg = joblib.load(os.path.join(model_path,alleles[0]+'.joblib'))
print(reg.predict(seqs[0]))
