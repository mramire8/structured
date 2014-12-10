from sklearn.datasets import base as bunch
from learner.strategy import Joint, Sequential
import numpy as np

def sample_data(data, train_idx, test_idx):
    sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch())
    
    if len(test_idx) > 0: #if there are test indexes
        sample.train.data = (np.array(data.data, dtype=object)[train_idx]).tolist()
        sample.test.data = data.data[test_idx]
        sample.train.target = data.target[train_idx]
        sample.test.target = data.target[test_idx]
        sample.target_names = data.target_names
    else:
        ## Just shuffle the data
        sample = data
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[train_idx]
        sample.train.data = data_lst.tolist()
        sample.train.target = data.train.target[train_idx]

    return sample.train, sample.test

def get_vectorizer(config):
    limit = config['limit']
    vectorizer = config['vectorizer']
    min_size = config['min_size']

    if vectorizer == 'tfidf':
        from sklearn.feature_extraction.text import  TfidfVectorizer
        return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))
    else:
        return None


def get_classifier(cl_name, **kwargs):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    clf = None
    if cl_name in "mnb":
        alpha = 1
        if 'parameter' in kwargs:
            alpha = kwargs['parameter']
        clf = MultinomialNB(alpha=alpha)
    elif cl_name == "lr":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegression(penalty="l1", C=c)
    elif cl_name == "lrl2":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegression(penalty="l2", C=c)
    else:
        raise ValueError("We need a classifier name for the student [lr|mnb]")
    return clf


def get_learner(config):
    from learner.base import Learner
    cl_name = config['model']
    clf = get_classifier(cl_name, parameter=config['parameter'])
    learner = Learner(clf)
    if config['type'] == 'joint':
        learner = Joint(clf, snippet_fn=None, utility_fn=None)
    elif config['type'] == 'sequential':
        learner = Sequential(clf, snippet_fn=None, utility_fn=None)
    else:
        raise ValueError("We don't know {} leaner".format(config['type']))
    learner.set_utility(config['utility'])
    learner.set_snippet_utility(config['snippet'])

    return learner


def get_expert(config):
    from expert.base import BaseExpert
    from expert.experts import PredictingExpert, SentenceExpert, TrueExpert
    cl_name = config['model']
    clf = get_classifier(cl_name, parameter=config['parameter'])
    
    expert = BaseExpert(clf)
    if config['type'] == 'true':
        expert = TrueExpert(None)
    elif config['type'] == 'pred':
        expert = PredictingExpert(clf)
    elif config['type'] == 'sent':
        tk = get_tokenizer(config['sent_tokenizer'])
        expert = SentenceExpert(clf, tokenizer=tk)
    else:
        raise Exception("We dont know {} expert".format(config['type']))

    return expert


def get_tokenizer(tk_name):
    if tk_name == 'nltk':
        import nltk
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return sent_detector
    else:
        raise Exception("Unknown sentence tokenizer")


def get_costfn(fn_name):
    if fn_name == 'unit':
        return unit_cost
    else:
        raise Exception("Unknown cost function")


def unit_cost(X):
    return X.shape[0]