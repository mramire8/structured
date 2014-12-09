
class TrueExpert(BaseExpert):
	"""docstring for TrueExpert"""
	def __init__(self, oracle):
		super(TrueExpert, self).__init__(oracle)
		
	def label(self, data, y=None):
		if 'target' in data:
			return data.target
		elif y is None:
			raise Exception("True labels are missing")
		else:
			return y

class PredictingExpert(BaseExpert):
	"""docstring for PredictingExpert"""
	def __init__(self, oracle):
		super(PredictingExpert, self).__init__(oracle)
		
	
	def label(self, data, y=None):
		return self.oracle.predict(data)

