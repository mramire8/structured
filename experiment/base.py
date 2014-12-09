__author__="mramire8"

import os, sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from sklearn import metrics
import utilities.experimentutils as exputil
import utilities.datautils as datautil
import utilities.configutils as cfgutil
from sklearn import cross_validation

class Experiment(object):
	"""Main experiment class to run according to configuration"""
	# def __init__(self, dataname, learner, expert, trials=5, folds=1, split=.5, costfn=None):
	def __init__(self, dataname, config, verbose=False):
		super(Experiment, self).__init__()
		self.verbose = verbose
		self.dataname = dataname
		self.config = config
		self.data 	= None
		self.trials	= None
		self.folds	= None
		self.split	= None
		self.costfn = None
		self.rnd_state = 32564
		self.vct = exputil.get_vectorizer(cfgutil.get_section_options(config, 'data'))

	def vectorize(self, data):
		data.train.bow = self.vct.fit_transform(data.train.data)
		data.test.bow = self.vct.transform(data.test.data)
		return data

	def cross_validation_data(data, **config):
		n = data.train.target.shape[0]
		cv= None

		if config['folds'] == 1 or 'test' not in data:
			cv = cross_validation.ShuffleSplit(n, n_iter=config['trials'], test_size=config['split'],
				random_state=self.rnd_state)
			config['folds'] = 1
		else: 
			cv = cross_validation.KFold(n, n_folds=config['folds'], random_state=self.rnd_state)
		return cv


	def start(self):
		trial = []
		self.data = datautil.load_dataset(self.dataname, categories=None, rnd=self.rnd_state, shuffle=True)
		self.data = self.vectorize(self.data)
		
		cv = get_cross_validation_data(self.data, cfgutil.get_section('experiment'))

		for train_index, text_index in cv:
			## get the data of this cv iteration
			train, test = exputil.sample_data(self.data, train_index, test_index)
			## get the expert and student
			learner = exputil.get_learner(cfgutil.get_config_section(config, 'learner'))
			expert = exputl.get_expert(cfgutil.get_config_section(config, 'expert'))
			expert.fit(train.bow, y=train.target)
			## do active learning
			results = main_loop(learner, expert, self.buget, self.bootstrap, train, test)
			
			## save the results
			trial.append(results)
		self.save_results(trial, self.dataname)

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

	def evaluate_oracle(self, query, labels):
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

