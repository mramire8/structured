__author__ = 'maru'

from expert.base import BaseExpert
import os

class AMTExpert(BaseExpert):

    def __init__(self, model):
        super(AMTExpert, self).__init__(model)

    def label(self, data, y=None):
        pass