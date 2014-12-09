import numpy as np

class RandomSampling(Learner):
	"""docstring for RandomSampling"""
	def __init__(self, arg):
		super(RandomSampling, self).__init__(model)
		

class ActiveLearner(Learner):
	"""docstring for ActiveLearner"""
	def __init__(self, model, utility=None):
		super(ActiveLearner, self).__init__(model)
		self.utility = self.utility_base
		self.rnd_state = np.random.RandomState(self.rnd_state)

	def utility_base(self, x):
		raise Exception("We need a utility function")

class StructuredLearer(ActiveLearner):
	"""docstring for StructuredLearer"""
	def __init__(self, model, snippet_model=None, utililty=None):
		super(StructuredLearer, self).__init__(model)
		self.snippet_model = snippet_model
		self.utility=utility

	def fit(self, X,y):
		#fit student
		
		#fit sentence
		pass

	def _utililty_rnd(X):
		return self.rnd_state.random()

	def _utility_unc(X):
		p = self.model.predict_proba(X)
		return 1. - p.max()

class Sequential(StructuredLearer):
	"""docstring for Sequential"""
	def __init__(self, arg):
		super(Sequential, self).__init__()
		self.arg = arg

	def next(pool, step):
		pass

class Joint(StructuredLearer):
	"""docstring for Joint"""
	def __init__(self, arg):
		super(Joint, self).__init__()
		self.arg = arg

	def next(pool, step):
		pass
		