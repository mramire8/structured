__author__ = 'maru'

import datautils as utils
from sklearn.datasets import base as bunch
from collections import defaultdict


def load_data_results(filename):

    import csv


    results = defaultdict(lambda: [])
    header = []
    with open(filename, 'rb') as csvfile:
        sents = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in sents:

            for k,v in row.items():
                results[k].append(v)

    return results


def load_amt(file_name):
    import numpy as np

    amt = load_data_results(file_name)

    ids = np.unique(amt['DOCID'])
    labels = np.array(amt['TARGET'])

    ordered = defaultdict(lambda: {})

    order = np.argsort(amt['ID'])
    print np.array(amt['ID'])[order]

    for i in order:
        ordered[amt['DOCID'][i]][amt['SENTID'][i][1:]] = amt['TEXT'][i]

    docs = []
    for docid in ids:
        ## Get the sentence of the document in order
        docs.append([ordered[docid][txt] for txt in sorted(ordered[docid], key=lambda x: int(x))])
    doc_text = [" THIS_IS_A_SEPARATOR ".join(s for s in docs)]
    sents = []
    sentid = amt['ID']
    sentlabels =amt['SENT_TARGET']


    return docs, ids, labels, sents, sentid, sentlabels

def load_amt_imdb(path, shuffle=True, rnd=2356, amt_labels=None):
    import numpy as np
    from collections import defaultdict


    data = utils.load_imdb(path, shuffle=shuffle, rnd=rnd)  # should brind data as is

    docs, ids, labels, sents, sentid, sentlabels = load_amt(amt_labels)

    data.train.data = docs
    data.train.target = labels
    data.train.docid = ids
    # get the document text
    # get the sentence labels (how?)
    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    # data = minimum_size(data, min_size=100)


    return data

