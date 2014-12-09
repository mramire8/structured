
class RandomSampling(Learner):
	"""docstring for RandomSampling"""
	def __init__(self, arg):
		super(RandomSampling, self).__init__()
		self.arg = arg

class ActiveLearner(Learner):
	"""docstring for ActiveLearner"""
	def __init__(self, arg):
		super(ActiveLearner, self).__init__()
		self.arg = arg

class StructuredLearer(ActiveLearner):
	"""docstring for StructuredLearer"""
	def __init__(self, arg):
		super(StructuredLearer, self).__init__()
		self.arg = arg


class Sequential(Learner):
	"""docstring for Sequential"""
	def __init__(self, arg):
		super(Sequential, self).__init__()
		self.arg = arg
		

class Joint(Learner):
	"""docstring for Joint"""
	def __init__(self, arg):
		super(Joint, self).__init__()
		self.arg = arg
		