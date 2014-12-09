
class TrueExpert(BaseExpert):
	"""docstring for TrueExpert"""
	def __init__(self, arg):
		super(TrueExpert, self).__init__()
		self.arg = arg
	def label(self, data, y=None):
		if y is None
			raise Exception("True labels are missing")
		return y

class PredictingExpert(BaseExpert):
	"""docstring for PredictingExpert"""
	def __init__(self, oracle):
		super(PredictingExpert, self).__init__()
		self.oracle = oracle
	
	def label(self, data, y=None):
		return self.oracle.predict(data)

