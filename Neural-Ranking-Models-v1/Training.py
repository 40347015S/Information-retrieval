import sys
sys.path.append("../Tools")
import numpy as np

#from sklearn import preprocessing
import ProcDoc
import RelPrep
import NRMprep


RES_POS = True
NUM_VOCAB = 51253

len_feats = 10
type_rank = "pointwise" # or pairwise
type_feat = "sparse"    # or embeddings
query_path = None
document_path = None
QDrel_file_path = None

corpus = "TDT2"

# qry and doc
if query_path == None: 
    query_path = "../Corpus/" + corpus + "/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
if document_path == None:
    document_path = "../Corpus/" + corpus + "/SPLIT_DOC_WDID_NEW"
if QDrel_file_path == None:
    QDrel_file_path = "../Significant-Words-Language-Models/train-qry-results-0.675969697596.txt"
# relevancy set
hmm_training_set = ProcDoc.readRELdict()
        
# read document, reserve position
doc = ProcDoc.readFile(document_path)
doc = ProcDoc.docPreproc(doc, RES_POS)
		
# read query, reserve position
qry = ProcDoc.readFile(query_path)
qry = ProcDoc.qryPreproc(qry, hmm_training_set, RES_POS)
QDrel =  RelPrep.readQDRel(QDrel_file_path)

print len(qry), len(doc)
print len(QDrel)
NRMprep.getTrainAndValidation(qry, doc, QDrel, NUM_VOCAB, type_rank, type_feat)
# (pointwise or pairwise) and (sparse or embeddings)
# prepare data and label
# NRMPrep.getTrainAndValidation(qry, doc, type_rank, type_feat, percent)
# return train.data, train.label, val.data, val.label

# create model
# {ScoreFunc, Rank, RankProb}.createModel(input_dim, type_rank)
# return model

# compile model and (fit or fit_generator)

# model.save 
