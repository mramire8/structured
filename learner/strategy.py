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

    def bootstrap(self, pool, step=2, shuffle=False):
        from collections import defaultdict

        step = int(step / 2)
        data = defaultdict(lambda: [])

        for i in pool.remaining:
            data[pool.target[i]].append(i)

        chosen = []
        for label in data.keys():
            candidates = data[label]
            if shuffle:
                indices = self.randgen.permutation(len(candidates))
            else:
                indices = range(len(candidates))
            chosen.extend([candidates[index] for index in indices[:step]])

        return chosen

class ActiveLearner(Learner):
    """docstring for ActiveLearner"""
    def __init__(self, model, utility=None, seed=54321):
        super(ActiveLearner, self).__init__(model, seed=seed)
        self.utility = self.utility_base
        self.rnd_state = np.random.RandomState(self.seed)

    def utility_base(self, x):
        raise Exception("We need a utility function")

class StructuredLearner(ActiveLearner):
    """docstring for StructuredLearner"""
    def __init__(self, model, snippet_fn=None, utility_fn=None):
        super(StructuredLearner, self).__init__(model)
        import copy
        self.snippet_model = copy.copy(model)
        self.utility=utility_fn
        self.snippet_utility = snippet_fn
        self.sent_tokenizer = None
        self.vct = None

    @staticmethod
    def convert_to_sentence(X_text, y, sent_tk, limit=None):
        sent_train = []
        labels = []
        ## Convert the documents into sentences: train
        print X_text[:3]
        for t, sentences in zip(y, sent_tk.batch_tokenize(X_text)):
            if limit > 0:
                sents = [s for s in sentences if len(s.strip()) > limit]
            elif limit == 0 or limit is None:
                sents = [s for s in sentences]
            sent_train.extend(sents)  # at the sentences separately as individual documents
            labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

        return sent_train, labels #, dump

    def fit(self, X,y, doc_text=None, limit=None):
        #fit student
        self.model.fit(X,y)
        #fit sentence
        sx, sy = self.convert_to_sentence(doc_text, y, self.sent_tokenizer, limit=limit)
        sx = self.vct.transform(sx)
        self.snippet_model.fit(sx, sy)
        
        return self

    def _utililty_rnd(self, X):
        if X.shape[0] ==1:
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
        raise NotImplementedError("Implement in the subclass")

    def _compute_snippet(self, X):
        raise NotImplementedError("Implement in the subclass")

    def _query(self, pool, indices):
        q = bunch.Bunch()
        q.data   = pool.bow[indices] 
        q.text   = pool.data[indices]
        q.target = pool.target[indices]
        q.index  = indices
        raise NotImplementedError("Implement in the subclass")

    def set_utility(self, util):
        if util == 'rnd':
            self.utility=self._utililty_rnd
        elif util=='unc':
            self.utility=self._utililty_unc

    def set_snippet_utility(self, util):
        if util == 'rnd':
            self.utility=self._snippet_rnd
        elif util=='max':
            self.utility=self._snnipet_max

    def set_sent_tokenizer(self, tokenizer):
        self.sent_tokenizer = tokenizer

    def set_vct(self, vct):
        self.vct = vct

    ## SNIPPET UTILITY FUNCTIONS
    def _snippet_max(self, X):
        p = self.snippet_model.predict_proba(X)
        if X.shape[0] ==1:
            return p.max()
        else:
            return p.max(axis=1)

    def _snippet_rnd(self, X):
        return self.rnd_state.random_sample(X.shape[0])

    def _snippet_first(self, X):
        raise NotImplementedError("first snippet utility is not impelemented yet")

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model, 
            self.snippet_model, self.utility, self.snippet_utility)


class Sequential(StructuredLearner):
    """docstring for Sequential"""
    def __init__(self, model, snippet_fn=None, utility_fn=None):
        super(Sequential, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn)

    def _subsample_pool(self, pool):
        subpool = list(pool.remaining)
        self.rnd_state.shuffle(subpool)
        subpool = subpool[:250]
        x = pool.bow[subpool]
        x_text = pool.data[subpool]
        return x, x_text, subpool

    def next(self, pool, step):
        x, x_text = self._subsample_pool()
        #compute utility
        utility = self._compute_utility(x)
        #compute best snippet
        snippet = self._compute_snippet(x, x_text)
        #select x, then s
        raise NotImplementedError("next sequential not implemented")


class Joint(StructuredLearner):
    """docstring for Joint"""
    def __init__(self, model, snippet_fn=None, utility_fn=None):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn)

    def _subsample_pool(self, pool):
        subpool = list(pool.remaining)
        self.rnd_state.shuffle(subpool)
        subpool = subpool[:250]
        x = pool.bow[subpool]
        x_text = pool.data[subpool]
        return x, x_text, subpool

    def next(self, pool, step):
        x, x_text, subpool = self._subsample_pool()

        #compute utlity
        utility = self.compute_utility(x)
        #comput best snippet
        snippet, snippet_text = self.compute_snippet(x, x_text)
        #multiply
        joint = utility * snippet 
        order = np.argsort(joint)[::-1]
        index = [subpool[i] for i in order[:step]]
        #build the query
        query = self._query(pool, index)
        return query
