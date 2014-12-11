import os
from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import base as bunch
import numpy as np


def keep_header_subject(text, keep_subject=False):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')

    sub = [l for l in _before.split("\n") if "Subject:" in l]
    if keep_subject:
        final = sub[0] + "\n" + after
    else:
        final = after
    return final


def load_imdb(path, subset="all", shuffle=True, rnd=2356):
    """
    load text files from IMDB movie reviews from folders to memory
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: ranom seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        data[subset] = load_files("{0}/{1}".format(path, subset), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    elif subset == "all":
        data["train"] = load_files("{0}/{1}".format(path, "train"), encoding="latin-1", load_content=True,
                                   random_state=rnd)
        data["test"] = load_files("{0}/{1}".format(path, "test"), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst.tolist()

    data = minimum_size(data)

    return data


def load_aviation(path, subset="all", shuffle=True, rnd=2356):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """
    # from sklearn.cross_validation import  ShuffleSplit
    data = bunch.Bunch()

    if subset in ('train', 'test'):
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(path, encoding="latin1", load_content=True,
                                   random_state=rnd)
        data.data = [keep_header_subject(text) for text in data.data]
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst.tolist()

    data = minimum_size(data)
    return data


def minimum_size(data):

    for part in data.keys():
        if len(data[part].data) != len(data[part].target):
            raise Exception("There is something wrong with the data")
        filtered = [(x, y) for x, y in zip(data[part].data, data[part].target) if len(x.strip()) >= 10]
        data[part].data = [x for x, _ in filtered]
        data[part].target = np.array([y for _, y in filtered])
    return data


def load_20newsgroups(category=None, shuffle=True, rnd=1):
    categories = {'religion': ['alt.atheism', 'talk.religion.misc'],
                  'graphics': ['comp.graphics', 'comp.windows.x'],
                  'hardware': ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  'baseball': ['rec.sport.baseball', 'sci.crypt']}
    cat = None
    if category is not None:
        cat = categories[category]
        
    data = bunch.Bunch()
    data.train = fetch_20newsgroups(subset='train', categories=cat, remove=('headers','footers', 'quotes'),
                                    shuffle=shuffle, random_state=rnd)

    data.train.data = [keep_header_subject(text) for text in data.train.data]

    data.test = fetch_20newsgroups(subset='test', categories=cat, remove=('headers','footers', 'quotes'),
                                   shuffle=shuffle, random_state=rnd)

    data.test.data = [keep_header_subject(text) for text in data.test.data]

    categories = data.train.target_names

    data = minimum_size(data)

    return data


def load_dataset(name, path, categories=None, rnd=2356, shuffle=True):
    data = bunch.Bunch()

    if "imdb" in name:
        ########## IMDB MOVIE REVIEWS ###########
        # data = bunch.Bunch(load_imdb(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size, fix_k=fixk, raw=raw))  # should brind data as is
        data = load_imdb(path, shuffle=shuffle, rnd=rnd)  # should brind data as is
    elif "aviation" in name:
        ########## sraa dataset ######
        data = load_aviation(path, shuffle=shuffle, rnd=rnd)
    elif "20news" in name:
        ########## 20 news groups ######
        data = load_20newsgroups(category=categories, shuffle=shuffle, rnd=rnd)
    else:
        raise Exception("We do not know {} dataset".format(name.upper()))

    return data