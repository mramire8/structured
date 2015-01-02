__author__ = 'maru'


class TwitterSentenceTokenizer(object):
    def __init__(self):
        pass

    def tokenize_sents(self, twitter_objs):
        return [sent.split("######") for sent in twitter_objs]

    def tokenize(self, twitter_objs):
        return twitter_objs.split("######")

    def __call__(self, twitter_objs):
        return twitter_objs

    def __str__(self):
        return self.__class__.__name__