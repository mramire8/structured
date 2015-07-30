__author__ = 'maru'

class AMTSentenceTokenizer(object):
    def __init__(self):
        pass

    def tokenize_sents(self, doc):
        return [sent.split("THIS_IS_A_SEPARATOR") for sent in doc]

    def tokenize(self, doc):
        return doc.split("THIS_IS_A_SEPARATOR")

    def __call__(self, doc):
        return doc

    def __str__(self):
        return self.__class__.__name__