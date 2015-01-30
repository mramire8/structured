__author__ = 'mramire8'
from strategy import Sequential, Joint


class JointSingleStudent(Sequential):
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(JointSingleStudent, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.snippet_model = self.model

    def fit(self, X, y, doc_text=None, limit=None):
        # fit student
        self.model.fit(X, y)
        # re-assign the snippet model
        self.snippet_model = self.model
        return self


class SequentialSingleStudent(Joint):
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(SequentialSingleStudent, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.snippet_model = self.model

    def fit(self, X, y, doc_text=None, limit=None):
        # fit student
        self.model.fit(X, y)
        # re-assign the snippet model
        self.snippet_model = self.model
        return self
