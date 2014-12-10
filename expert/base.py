
class BaseExpert(object):
    """docstring for BaseExpert"""
    def __init__(self, model):
        super(BaseExpert, self).__init__()
        self.model = model

    def label(self, data, y=None):
        raise Exception("Expert has not model")        

    def fit(self, data, y=None):
        if y is not None:
            self.model.fit(data.bow, y)
        else:
            self.model.fit(data.bow, data.target)
        return self