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

            for k, v in row.items():
                results[k].append(v)

    return results


def load_amt(path, file_name):
    '''
    Load AMT IMDB dataset
    :param file_name:
    :return:
    '''
    import numpy as np

    amt = load_data_results(path + "/" +'amt.results.csv')
    sent_target = load_data_results(path + "/" +'amt_lbl_neucoin.csv')
    # Document id
    ids = np.unique(amt['DOCID'])

    # Document target
    labels = np.array(amt['TARGET'])

    ordered = defaultdict(lambda: {})

    # document and sentence id
    order = np.argsort(amt['ID'])
    print np.array(amt['ID'])[order]

    for i in order:
        ordered[amt['DOCID'][i]][amt['SENTID'][i][1:]] = amt['TEXT'][i]

    docs = []
    for docid in ids:
        #  Get the sentence of the document in order of sentence
        docs.append([ordered[docid][txt] for txt in sorted(ordered[docid], key=lambda x: int(x))])

    # Convert into documents with separator
    doc_text = ["THIS_IS_A_SEPARATOR".join(s) for s in docs]
    sents = docs

    sentid = amt['ID']
    sentlabels = []

    # for every document
    for docid in ids:
        # Get the sentence of the document in order
        doc_sent = []
        for s in ordered[docid].keys():
            sloc = "{}S{}".format(docid,s)
            sid = sent_target['ID'].index(sloc)
            doc_sent.append(sent_target['TARGET'][sid])
        sentlabels.append(doc_sent)
        # sentlabels.append([ordered[docid][txt] for txt in sorted(ordered[docid], key=lambda x: int(x))])

    return docs, ids, labels, sents, sentid, sentlabels


def load_amt_imdb(path, shuffle=True, rnd=2356, amt_labels=None):
    '''
    Load AMT imdb data, imdb original data and load labels
    :param path:
    :param shuffle:
    :param rnd:
    :param amt_labels:
    :return:
    '''
    import numpy as np
    from collections import defaultdict

    # should brind data as is, we will use vct and test data from here

    data = utils.load_imdb(path, shuffle=shuffle, rnd=rnd)

    docs, ids, labels, sents, sentid, sentlabels = load_amt(path, amt_labels)

    data.train.data = ["THIS_IS_A_SEPARATOR".join(d) for d in docs]
    # data.train.target = labels
    data.train.target = np.array(sentlabels, dtype=object)
    data.train.docid = ids
    # get the document text
    # get the sentence labels (how?)
    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(len(data.train.target))
        random_state.shuffle(indices)
        # data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    # data = minimum_size(data, min_size=100)

    return data
