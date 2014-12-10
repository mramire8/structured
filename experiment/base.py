__author__="mramire8"

import os, sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from sklearn import metrics
import utilities.experimentutils as exputil
import utilities.datautils as datautil
import utilities.configutils as cfgutil
from sklearn import cross_validation
import numpy as np
from collections import defaultdict
from learner.strategy import BootstrapFromEach
from sklearn.datasets import base as bunch

class Experiment(object):
    """Main experiment class to run according to configuration"""
    # def __init__(self, dataname, learner, expert, trials=5, folds=1, split=.5, costfn=None):
    def __init__(self, dataname, config, verbose=False):
        super(Experiment, self).__init__()
        self.verbose = verbose
        self.dataname = dataname
        self.data_cat = None
        self.config = config
        self.data     = None
        self.trials    = None
        self.folds    = None
        self.split    = None
        self.costfn = None
        self.budget = None
        self.max_iteration = None
        self.step = None
        self.rnd_state = np.random.RandomState(32564)
        self.remaining = None
        self.vct = exputil.get_vectorizer(cfgutil.get_section_options(config, 'data'))

    def vectorize(self, data):
        data.train.bow = self.vct.fit_transform(data.train.data)
        data.test.bow = self.vct.transform(data.test.data)
        return data

    def cross_validation_data(self, data, **config):
        n = data.train.target.shape[0]
        cv= None

        if config['folds'] == 1 and 'test' not in data.keys():
            cv = cross_validation.ShuffleSplit(n, n_iter=config['trials'], test_size=config['split'],
                random_state=self.rnd_state)
            config['folds'] = 1
        elif 'test' in data.keys():
            cv = cross_validation.ShuffleSplit(n, n_iter=config['trials'], test_size=0.0,
                random_state=self.rnd_state)
            config['folds'] = 1

        else: 
            cv = cross_validation.KFold(n, n_folds=config['folds'], random_state=self.rnd_state)
        return cv

    def _setup_options(self, config_obj):
        config = cfgutil.get_section_options(config_obj, 'experiment')
        self.trials = config['trials']
        self.folds = config['folds']
        self.max_iteration = config['maxiter']
        self.step     = config['stepsize']
        self.budget     = config['budget']
        self.prefix = config['fileprefix']
        self.output = config['outputdir']
        self.costfn = exputil.get_costfn(config['costfunction'])
        config = cfgutil.get_section_options(config_obj, 'data')
        self.split = config['split']
        self.data_cat = config['categories']
        self.limit = config['limit']

    def start(self):

        trial = []
        self._setup_options(self.config)
        self.data = datautil.load_dataset(self.dataname, categories=self.data_cat, rnd=self.rnd_state, shuffle=True)
        self.data = self.vectorize(self.data)
        cv = self.cross_validation_data(self.data,folds=self.folds, trials=self.trials, split=self.split)
        
        for train_index, test_index in cv:
            ## get the data of this cv iteration
            train, test = exputil.sample_data(self.data, train_index, test_index)
            ## get the expert and student
            learner = exputil.get_learner(cfgutil.get_config_section(self.config, 'learner'))
            expert = exputil.get_expert(cfgutil.get_config_section(self.config, 'expert'))
            expert.fit(train.bow, y=train.target, vct=self.vct)
            ## do active learning
            results = self.main_loop(learner, expert, self.budget, self.bootstrap, train, test)
            
            ## save the results
            trial.append(results)
        self.save_results(trial, self.dataname)

    def bootstrap(self, pool, bt):
        initial = BootstrapFromEach.bootstrap(pool, k=bt, shuffle=False)
        bootstrap = bunch.Buch()
        bootstrap.index = initial
        bootstrap.bow = pool.bow[initial]
        bootstrap.data= pool.data[initial]
        bootstrap.target = pool.target[initial]

        return bootstrap

    def update_cost(self, current_cost, query):
        return current_cost + self.costfn(query)

    def evaluate(self, learner, test):
        prediction = learner.predict(test.bow)
        pred_proba = learner.predict_proba(test.bow)
        accu = metrics.accuracy_score(test.target, prediction)
        auc = metrics.roc_auc_score(test.target, pred_proba)
        return {'auc':auc, 'accu':accu}

    def evaluate_oracle(self, query, predictions, labels=None):
        cm = metrics.confusion_matrix(query.target, predictions, labels=labels)
        return cm

    def update_run_results(self, results, step, oracle, iteration):
        results['accuracy'][iteration].append(step['accu'])
        results['auc'][iteration].append(step['auc'])
        results['ora_accu'][iteration].append(oracle)
        return results

    def update_pool(self, pool, query):
        ## remove from remaining
        for q in query.index:
            pool.remaining.remove(q)
        return pool

    def main_loop(self, learner, expert, budget, bootstrap, pool, test):
        from  collections import deque
        iteration = 0
        current_cost = 0
        rnd_set = range(pool.target.shape[0])
        self.rnd_state.shuffle(rnd_set)
        remaining = deque(rnd_set)
        pool.remaining = remaining
        results = self._start_results()
        while current_cost <= budget and iteration < self.max_iteration:
            if iteration == 0:
                #bootstrap
                bt = self.bootstrap(pool, bootstrap)
                learner.fit(bt)
                pass
            else:
                query = learner.next(self.step, pool)
                labels = expert.label(query)
                pool = self.update_pool(pool, query)
                current_cost = self.update_cost(current_cost, query)
                learner.fit(query, labels,)
                step_results = self.evaluate(learner, test)
                step_oracle = self.evaluate_oracle(query, labels)
                results = self.update_run_results(results, step_results, step_oracle, current_cost)
            iteration +=1
        return results

    def _start_results(self):
        r = {}
        r['accuracy']     = defaultdict(lambda: [])
        r['auc']        = defaultdict(lambda: [])
        r['ora_accu']    = defaultdict(lambda: [])
        
        return r

    def report_results(self, results):
        pass

