import os

def readQDRel(QDrel_file_path):          #return relevance_score {qry1:{doc3:7.733, doc1:7.812}, qry2:[...], ...}
    with open(QDrel_file_path, 'r') as f:
        QDrel = {}
        for line in f.readlines():
            if line != '\n':
                line = line.split()
                if len(line) == 1:
                    query_name = line[0]
                    QDrel[query_name] = {}
                    continue
                QDrel[query_name][line[0]] = line[1]
    return QDrel

