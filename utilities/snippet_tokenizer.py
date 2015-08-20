__author__ = 'maru'

import numpy as np

class SnippetTokenizer(object):

    def __init__(self, k=(1,1)):
        import nltk
        self.sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
        self.k = k
        self.separator = " "
        self.split_bound = '\\b\\w+\\b'

    def tokenize_sents(self, doc):
        return [self.tokenize(sent) for sent in doc]

    def tokenize(self, doc):
        return self.get_sentences(doc, self.k)

    def set_snippet_size(self, k):
        self.k = k

    def __str__(self):
        return self.__class__.__name__

    def get_sentences_k(self, sentences, k):
        import itertools as it

        all_sents = []
        n = min(len(sentences[:30]), k[1]+1)
        for i in range(k[0],n):
            pairs = it.combinations(sentences[:30], i)
            all_sents.extend([self.separator.join(p) for p in pairs])
        return all_sents

    def get_sentences(self, doc, k):

        d_sent = self.sent_tk.tokenize_sents([doc])

        return self.get_sentences_k(d_sent[0], k)


class First1SnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1)):
        super(First1SnippetTokenizer,self).__init__(k)

    def get_sentences_k(self, sentences, k):
        import itertools as it
        n = len(sentences[:30])
        all_sents = []
        pairs = it.combinations(sentences[:30], min(k[1], n))
        for p in pairs:
            all_sents.append(self.separator.join(p))
            break
        return all_sents


class Random1SnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1), seed=5432):
        super(Random1SnippetTokenizer,self).__init__(k)
        self.rnd = np.random.RandomState(seed)

    def get_sentences_k(self, sentences, k):
        import itertools as it
        n = len(sentences[:30])
        all_sents = []

        pairs = it.combinations(sentences[:30], min(k[1], n))
        for p in pairs:
            all_sents.append(self.separator.join(p))

        pick = self.rnd.randint(0,len(all_sents),1)
        return [all_sents[pick]]
