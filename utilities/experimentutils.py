from sklearn.datasets.base import Bunch
from sklearn.datasets import base as bunch


def sample_data(data, train_idx, test_idx):
    sample = Bunch()
    sample.train.data = data.data[train_idx]
    sample.test.data = data.data[test_idx]
    sample.train.target = data.target[train_idx]
    sample.test.target = data.target[test_idx]
    sample.target_names = data.target_names

return sample

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
    elif cl_name == "lradapt":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegressionAdaptive(penalty="l1", C=c)
    elif cl_name == "lradaptv2":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegressionAdaptiveV2(penalty="l1", C=c)
    else:
        raise ValueError("We need a classifier name for the student [lr|mnb]")
    return clf


def get_learner(config):
	from learner.base import Learner
	cl_name = config['model']
	clf = get_classifier(cl_name, parameter=config['parameter'])
	learner = Learner(clf)
	if config['type'] == 'joint':
		learner = SequentialLearner
	elif config['type'] == 'sequential':
	else:
		raise ValueError("We don't know {} leaner".format(config['type']))
    return learner

def get_expert(config):
    from expert.base import BaseExpert
	cl_name = config['model']
	clf = get_classifier(cl_name, parameter=config['parameter'])
	learner = BaseExpert(clf)
