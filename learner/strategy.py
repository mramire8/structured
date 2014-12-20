import numpy as np
from base import Learner
from sklearn.datasets import base as bunch


class RandomSampling(Learner):
    """docstring for RandomSampling"""

    def __init__(self, model):
        super(RandomSampling, self).__init__(model)


class BootstrapFromEach(Learner):
    def __init__(self, model, seed=None):
        super(BootstrapFromEach, self).__init__(model, seed=seed)
        self.rnd_state = np.random.RandomState(self.seed)

    def bootstrap(self, pool, step=2, shuffle=False):
        """
        bootstrap by selecting step/2 instances per class, in a binary dataset
        :param pool: bunch containing the available data
            pool contains:
                target: true labels of the examples
                ramaining: list of available examples in the pool to use
        :param step: how many examples to select in total
        :param shuffle: shuffle the data before selecting or not (important for sequential methods)
        :return: list of indices of selected examples
        """
        from collections import defaultdict

        step = int(step / 2)
        data = defaultdict(lambda: [])

        for i in pool.remaining:
            data[pool.target[i]].append(i)

        chosen = []
        for label in data.keys():
            candidates = data[label]
            if shuffle:
                indices = self.rnd_state.permutation(len(candidates))
            else:
                indices = range(len(candidates))
            chosen.extend([candidates[index] for index in indices[:step]])

        return chosen


class ActiveLearner(Learner):
    """ActiveLearner class that defines a simple utility based pool sampling strategy"""

    def __init__(self, model, utility=None, seed=1):
        super(ActiveLearner, self).__init__(model, seed=seed)
        self.utility = self.utility_base
        self.rnd_state = np.random.RandomState(self.seed)

    def utility_base(self, x):
        raise Exception("We need a utility function")


class StructuredLearner(ActiveLearner):
    """StructuredLearner is the Structured reading implementation """
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(StructuredLearner, self).__init__(model, seed=seed)
        import copy

        self.snippet_model = copy.copy(model)
        self.utility = utility_fn
        self.snippet_utility = snippet_fn
        self.sent_tokenizer = None
        self.vct = None
        self.calibrate = None
        self.sent_rnd = np.random.RandomState(self.seed)

    @staticmethod
    def convert_to_sentence(X_text, y, sent_tk, limit=None):
        """
        >>> import nltk
        >>> sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
        >>> print StructuredLearner.convert_to_sentence(['hi there. you rock. y!'], [1], sent_tk, limit=2)
        (['hi there.', 'you rock.'], [1, 1])
        """
        sent_train = []
        labels = []

        ## Convert the documents into sentences: train
        # for t, sentences in zip(y, sent_tk.batch_tokenize(X_text)):
        for t, sentences in zip(y, sent_tk.tokenize_sents(X_text)):
            if limit > 0:
                sents = [s for s in sentences if len(s.strip()) > limit]
            elif limit == 0 or limit is None:
                sents = [s for s in sentences]
            sent_train.extend(sents)  # at the sentences separately as individual documents
            labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

        return sent_train, labels  # , dump

    def get_name(self):
        return "{}{}".format(self.utility.__name__, self.snippet_utility.__name__)

    def fit(self, X, y, doc_text=None, limit=None):
        # fit student
        self.model.fit(X, y)
        #fit sentence
        sx, sy = self.convert_to_sentence(doc_text, y, self.sent_tokenizer, limit=limit)
        sx = self.vct.transform(sx)
        self.snippet_model.fit(sx, sy)

        return self

    def _utility_rnd(self, X):
        if X.shape[0] == 1:
            return self.rnd_state.random_sample()
        else:
            return self.rnd_state.random_sample(X.shape[0])

    def _utility_unc(self, X):
        p = self.model.predict_proba(X)
        if X.shape[0] == 1:
            return 1. - p.max()
        else:
            return 1. - p.max(axis=1)

    def _subsample_pool(self, X):
        raise NotImplementedError("Implement in the subclass")

    def _compute_utility(self, X):
        return self.utility(X)

    def _query(self, pool, snippets, indices):
        q = bunch.Bunch()
        q.data = pool.bow[indices]
        q.bow = self.vct.transform(snippets)
        q.text = pool.data[indices]
        q.target = pool.target[indices]
        q.snippet = snippets
        q.index = indices
        return q

    def _do_calibration(self, scores, y_pred):
        """
        perform calibration on the scores per sentence
        :param socres: scores per document sentences as computed by  _Snippet_fn
        :param y_pred: prediction of the sentences as predicted by $P_S$. This predictions are shift by adding +1
            thus class 0 appears as 1, class 1 as 2, and so on.
        :raise NotImplementedError:
        """
        raise NotImplementedError("This method should be assigned from configuration")

    def zscores_pred(self, scores, y_pred):
        # if self.calibrate:
        from sklearn import preprocessing
        # prediction +1 to preserve the spcarcity of the matrix
        c0_scores = scores[y_pred == (0+1)]
        c1_scores = scores[y_pred == (1+1)]
        c0_scores = preprocessing.scale(c0_scores)
        c1_scores = preprocessing.scale(c1_scores)
        scores[y_pred == (0+1)] = c0_scores
        scores[y_pred == (1+1)] = c1_scores
        return scores
        # else:
        #     return scores
    
    def zscores_rank(self, scores, y_pred):
        
        from sklearn import preprocessing
        # prediction +1 to preserve the spcarcity of the matrix
        _scores = np.array(scores)
        _scores[y_pred == 1] = scores[y_pred == 1]
        _scores[y_pred == 2] = 1. - scores[y_pred == 2]
        
        median_score = np.median(_scores[y_pred>0])
        
        rank1_indices = np.bitwise_and(_scores > median_score, y_pred > 0)
        rank2_indices = np.bitwise_and(_scores <= median_score, y_pred > 0)
        
        _scores[rank1_indices] = preprocessing.scale(_scores[rank1_indices])
        _scores[rank2_indices] = preprocessing.scale(1. - _scores[rank2_indices])
        
        return _scores

    def _no_calibrate(self, scores, y_pred):
        return scores

    def set_utility(self, util):
        if util == 'rnd':
            self.utility = self._utility_rnd
        elif util == 'unc':
            self.utility = self._utility_unc

    def set_snippet_utility(self, util):
        if util == 'rnd':
            self.snippet_utility = self._snippet_rnd
        elif util == 'sr':
            self.snippet_utility = self._snippet_max
        elif util == 'first1' or util == 'true':
            self.snippet_utility = self._snippet_first

    def set_calibration_method(self, cal_name):
        self._do_calibration = getattr(self, cal_name)


    def set_sent_tokenizer(self, tokenizer):
        self.sent_tokenizer = tokenizer

    def set_vct(self, vct):
        self.vct = vct

    def set_calibration(self, cali):
        self.calibrate = cali

    ## SNIPPET UTILITY FUNCTIONS
    def _snippet_max(self, X):
        p = self.snippet_model.predict_proba(X)
        if X.shape[0] == 1:
            return p.max()
        else:
            return p.max(axis=1)

    def _snippet_rnd(self, X):
        return self.sent_rnd.random_sample(X.shape[0])

    def _snippet_first(self, X):
        n = X.shape[0]
        scores = np.zeros(n)
        scores[0] = 1
        return scores

    def _create_matrix(self, x_sent, x_len):
        from scipy.sparse import lil_matrix

        # X = lil_matrix((len(x_sent), x_len))
        X = np.zeros((len(x_sent), x_len))

        return X

    def _get_sentences(self, x_text):
        text = self.sent_tokenizer.tokenize_sents(x_text)
        text_min = []
        for sentences in text:
            text_min.append([s for s in sentences if len(s.strip()) > 2])  # at least 2 characters
        return text_min

    def _compute_snippet(self, x_text):
        """select the sentence with the best score for each document"""
        # scores = super(Joint, self)._compute_snippet(x_text)
        import sys
        x_sent_bow = []
        x_len = 0
        x_sent = self._get_sentences(x_text)
        for sentences in x_sent:
            x_sent_bow.append(self.vct.transform(sentences))
            x_len = max(len(sentences), x_len)

        x_scores = self._create_matrix(x_sent, x_len)
        y_pred = self._create_matrix(x_sent, x_len)

        for i, s in enumerate(x_sent_bow):
            score_i = np.ones(x_len) * -1 * sys.maxint
            y_pred_i = np.zeros(x_len)
            score_i[:s.shape[0]] = self.snippet_utility(s)
            y_pred_i[:s.shape[0]] = self.snippet_model.predict(s) + 1  # add 1 to avoid prediction 0, keep the sparsity
            x_scores[i] = score_i
            y_pred[i] = y_pred_i

        x_scores = self._do_calibration(x_scores, y_pred)

        #Note: this works only if the max score is always > 0
        sent_index = x_scores.argmax(axis=1)  ## within each document thesentence with the max score
        # sent_index = np.array(sent_index.reshape(sent_index.shape[0]))[0] ## reshape, when sparse matrix
        sent_max = x_scores.max(axis=1)  ## within each document thesentence with the max score
        sent_text = [x_sent[i][maxx] for i, maxx in enumerate(sent_index)]
        sent_text = np.array(sent_text, dtype=object)
        return sent_max, sent_text

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model,
                                                                               self.snippet_model, self.utility,
                                                                               self.snippet_utility)


class Sequential(StructuredLearner):
    """docstring for Sequential"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(Sequential, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)

    def _subsample_pool(self, pool):
        subpool = list(pool.remaining)
        subpool = subpool[:250]
        x = pool.bow[subpool]
        x_text = pool.data[subpool]
        return x, x_text, subpool

    def next(self, pool, step):
        x, x_text, subpool = self._subsample_pool(pool)

        # compute utility
        utility = self._compute_utility(x)

        #compute best snippet
        snippet, snippet_text = self._compute_snippet(x_text)

        #select x, then s
        seq = utility
        order = np.argsort(seq)[::-1]
        index = [subpool[i] for i in order[:step]]

        query = self._query(pool, snippet_text[order][:step], index)
        return query


class Joint(StructuredLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)

    def _subsample_pool(self, pool):
        subpool = list(pool.remaining)
        self.rnd_state.shuffle(subpool)
        subpool = subpool[:250]
        x = pool.bow[subpool]
        x_text = pool.data[subpool]
        return x, x_text, subpool

    def next(self, pool, step):
        x, x_text, subpool = self._subsample_pool(pool)

        # compute utlity
        utility = self._compute_utility(x)

        #comput best snippet
        snippet, snippet_text = self._compute_snippet(x_text)

        #multiply
        joint = utility * snippet
        order = np.argsort(joint)[::-1]
        index = [subpool[i] for i in order[:step]]

        #build the query
        query = self._query(pool, snippet_text[order][:step], index)
        return query
