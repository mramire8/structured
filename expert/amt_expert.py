__author__ = 'maru'

from expert.base import BaseExpert
import os

class AMTExpert(BaseExpert):

    def __init__(self, model):
        super(AMTExpert, self).__init__(model)

    def _convert_label(self, l):
        if int(l) > 1:
            return None
        else:
            return int(l)

    def _convert_labels(self, l):
        return [self._convert_label(ll) for ll in l]

    def label(self, data, y=None):

        # return [self._convert_labels(yy) for yy in y]
        return self._convert_labels(y)

    def fit(self, X, y=None, vct=None):
        return self
