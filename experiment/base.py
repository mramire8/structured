__author__="mramire8"

from sklear import metrics
from utilities.experimentutils import load
class Experiment(object):
	"""Main experiment class to run according to configuration"""
	def __init__(self, dataname, learner, expert, trials=5, folds=1, split=.5, costfn=None):
		super(Experiment, self).__init__()
		self.dataname = dataname
		self.trials = trials
		self.folds = folds
		self.learner = learner
		self.expert = expert
		self.split = split
		self.costfn = costfn

	def vectorize(self, data):
		pass 

	def start():
		self.data = experimientutils.load(self.dataname)
		self.data = vetorize(self.data)
		for t in range(self.trials):
			train, test = split_data(self.data, t, rnd)
			learner = get_learner()
			expert = get_experter()
			results = main_loop(learner, expert, self.buget, self.bootstrap, train, test)
			update_trial_results(results)

	def bootstrap(self, bt):
		pass

	def update_cost(self, current_cost, query):
		return current_cost + self.costfn(query)

	def evaluate(self, learner, test):
		prediction = learner.predict(test.bow)
		pred_proba = learner.predict_proba(test.bow)
		accu = metrics.accuracy_score(test.target, prediction)
		auc = metrics.roc_auc_score(test.target, predict_proba)
		return {'auc':auc, 'accu':accu}

	def evaluate_oracle(self, query,labels):
		cm = metrics.confusion_matrix(query.target, labels)
		return cm

	def update_run_results(self, step, oracle, iteration):
		pass

	def main_loop(self, learner, expert, budget, bootstrap, pool, test):
		iteration = 0
		current_cost = 0
		while current_cost <= budget and iteration < self.max_iteration:
			if iteration == 0:
				#bootstrap
				bt = self.bootstrap(bootstrap)
				learner.fit(bt)
				pass
			else:
				query = learner.next(self.step, pool)
				labels = expert.label(query)
				pool = update_pool(pool, query)
				current_cost = update_cost(current_cost, query)
				learner.fit(query, labels)
				step_results = evaluate(learner, test)
				step_oracle = evaluate_oracle(query, labels)
				results = update_run_results(step_results, step_oracle, iteration)
			iteration +=1
		return results

	def report_results(self, results):
		pass

