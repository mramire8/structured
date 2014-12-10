import numpy as np
from base import Learner

class RandomSampling(Learner):
    """docstring for RandomSampling"""
    def __init__(self, arg):
        super(RandomSampling, self).__init__(model)


        
class BootstrapFromEach(Learner):
    def __init__(self, seed):
        super(BootstrapFromEach, self).__init__(seed=seed)

    def bootstrap(self, pool, k=2, shuffle=False):
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
    def __init__(self, model, utility=None):
        super(ActiveLearner, self).__init__(model)
        self.utility = self.utility_base
        self.rnd_state = np.random.RandomState(self.rnd_state)

    def utility_base(self, x):
        raise Exception("We need a utility function")

class StructuredLearer(ActiveLearner):
    """docstring for StructuredLearer"""
    def __init__(self, model, snippet_fn=None, utililty_fn=None):
        super(StructuredLearer, self).__init__(model)
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

    def _utility_unc(X):
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

    def _snippet_max(X):
        p = self.snippet_model.predict_proba(X)
        if X.shape[0] ==1:
            return p.max()
        else:
            return p.max(axis=1)

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model, 
            self.snippet_model, self.utility, self.snippet_utility)

class Sequential(StructuredLearer):
    """docstring for Sequential"""
    def __init__(self, arg):
        super(Sequential, self).__init__()
        self.arg = arg

    def next(pool, step):
        pass

class Joint(StructuredLearer):
    """docstring for Joint"""
    def __init__(self, arg):
        super(Joint, self).__init__()
        self.arg = arg

    def next(pool, step):
        pass
        