import numpy as np
from base import Learner

class RandomSampling(Learner):
    """docstring for RandomSampling"""
    def __init__(self, arg):
        super(RandomSampling, self).__init__(model)


        
class BootstrapFromEach(Learner):
    def __init__(self, model, seed):
        super(BootstrapFromEach, self).__init__(model, seed=seed)

    def bootstrap(self, pool, k=2, shuffle=False):
        from collections import defaultdict
        k = int(k / 2)
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
            chosen.extend([candidates[index] for index in indices[:k]])

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

    def fit(self, X,y):
        #fit student

        #fit sentence
        pass

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

    def set_utility(self, util):
        if util == 'rnd':
            self.utility=self._utililty_rnd
        elif util=='unc':
            self.utility=self._utililty_unc

    def set_snippet_utility(self, util):
        if util == 'rnd':
            self.utility=self._utililty_rnd
        elif util=='max':
            self.utility=self._snnipet_max

    def _snippet_max(self, X):
        p = self.snippet_model.predict_proba(X)
        if X.shape[0] ==1:
            return p.max()
        else:
            return p.max(axis=1)

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model, 
            self.snippet_model, self.utility, self.snippet_utility)


class Sequential(StructuredLearner):
    """docstring for Sequential"""
    def __init__(self, model, snippet_fn=None, utility_fn=None):
        super(Sequential, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn)

    def next(self, pool, step):
        pass


class Joint(StructuredLearner):
    """docstring for Joint"""
    def __init__(self, model, snippet_fn=None, utility_fn=None):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn)

    def next(self, pool, step):
        pass

