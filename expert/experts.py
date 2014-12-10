from base import BaseExpert

class TrueExpert(BaseExpert):
    """docstring for TrueExpert"""
    def __init__(self, oracle):
        super(TrueExpert, self).__init__(oracle)
        
    def label(self, data, y=None):
        if 'target' in data.keys():
            return data.target
        elif y is None:
            raise Exception("True labels are missing")
        else:
            return y
    def fit(self, X, y=None):
    	return self

class PredictingExpert(BaseExpert):
    """docstring for PredictingExpert"""
    def __init__(self, oracle):
        super(PredictingExpert, self).__init__(oracle)
        
    
    def label(self, data, y=None):
        return self.oracle.predict(data)

    def fit(self, X, y=None):
        if y is not None:
            self.oracle.fit(data.bow, y)
        else:
            self.oracle.fit(data.bow, data.target)
        return self

class SentenceExpert(PredictingExpert):
    """docstring for SentenceExpert"""
    def __init__(self, oracle, tokenizer=None):
        super(SentenceExpert, self).__init__(oracle)
        self.tokenizer = None

    def convert_to_sentence(self, X, y, vct, limit=None):
        sent_train = []
        labels = []
        tokenizer = vct.build_tokenizer()
        ## Convert the documents into sentences: train
        for t, sentences in zip(y, self.tokenizer.batch_tokenize(X)):

            if limit is None:
                sents = [s for s in sentences if len(tokenizer(s)) > 1]
            elif limit > 0:
                sents = [s for s in sentences if len(s.strip()) > limit]
            elif limit == 0:
                sents = [s for s in sentences]
            sent_train.extend(sents)  # at the sentences separately as individual documents
            labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

        return labels, sent_train #, dump

    def fit(self, X_text, y=None, vct=None):
    	sx, sy = self.convert_to_sentence(X_text, y, vct)
        self.oracle.fit(vct.transform(sx),sy)
        return self
        
