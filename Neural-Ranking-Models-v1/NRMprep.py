import numpy as np

def getTrainAndValidation(qry, doc, QDrel, num_vocab, type_rank, type_feat, percent = 100):
    train={"data":[], "label":[]}
    if type_rank == 'pointwise':
        if type_feat == 'sparse':
            tfvc = np.zeros(num_vocab)
            for qry_filename in QDrel:
                tfvq = np.zeros(num_vocab)
                for word in qry[qry_filename]:
                    tfvq[word] += 1
                for doc_filename in QDrel[qry_filename]:
                    a_data = np.append(tfvc, tfvq)
                    tfvd = np.zeros(num_vocab)
                    for word in doc[doc_filename]:
                        tfvd[word] += 1
                    a_data = np.append(a_data, tfvd)
                    train["data"].append(a_data)
                    train["label"].append(QDrel[qry_filename][doc_filename])
            np.array(train["data"])
            np.array(train["label"])
            return train
        #if type_feat == 'dense':
            
