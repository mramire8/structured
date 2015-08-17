__author__ = 'maru'

__author__ = 'maru'


class SnippetTokenizer(object):

    def __init__(self, k=1):
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

    def __call__(self, doc):
        return doc

    def __str__(self):
        return self.__class__.__name__

    def get_sentences_k(self, sentences, k):
        import itertools as it

        all_sents = []
        for i in range(1,k+1):
            pairs = it.combinations(sentences, i)
            all_sents.extend([self.separator.join(p) for p in pairs])
        return all_sents

    def get_sentences(self, doc, k):

        d_sent = self.sent_tk.tokenize_sents([doc])

        return self.get_sentences_k(d_sent[0], k)
