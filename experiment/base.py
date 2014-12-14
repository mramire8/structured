__author__ = "mramire8"

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
    def __init__(self, dataname, config, verbose=False, debug=False):
        super(Experiment, self).__init__()
        self.verbose = verbose
        self.debug = debug
        self.config = config

        self.dataname = dataname
        self.data_cat = None
        self.data = None
        self.data_path = None

        self.trials = None
        self.folds = None
        self.split = None
        self.costfn = None
        self.budget = None
        self.max_iteration = None
        self.step = None
        self.bootstrap_size = None
        self.seed = None
        self.output = None

        self.rnd_state = np.random.RandomState(32564)
        self.remaining = None
        self.vct = exputil.get_vectorizer(cfgutil.get_section_options(config, 'data'))
        self.sent_tokenizer = None

    def vectorize(self, data):
        data.train.bow = self.vct.fit_transform(data.train.data)
        data.test.bow = self.vct.transform(data.test.data)
        return data

    def cross_validation_data(self, data, **config):
        n = data.train.target.shape[0]
        cv = None

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

        # experiment related config
        config = cfgutil.get_section_options(config_obj, 'experiment')
        self.trials = config['trials']
        self.folds = config['folds']
        self.max_iteration = config['maxiter']
        self.step = config['stepsize']
        self.budget = config['budget']
        self.prefix = config['fileprefix']
        self.output = config['outputdir']
        self.seed = config['seed']
        self.bootstrap_size = config['bootstrap']
        self.costfn = exputil.get_costfn(config['costfunction'])

        #data related config
        config = cfgutil.get_section_options(config_obj, 'data')
        self.split = config['split']
        self.data_cat = config['categories']
        self.limit = config['limit']
        self.data_path = config['path']

        #data related config
        config = cfgutil.get_section_options(config_obj, 'expert')
        self.sent_tokenizer = exputil.get_tokenizer(config['sent_tokenizer'])

    def start(self):
        print self.get_name()
        trial = []
        self._setup_options(self.config)
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed, shuffle=True)
        self.data = self.vectorize(self.data)
        cv = self.cross_validation_data(self.data, folds=self.folds, trials=self.trials, split=self.split)
        t = 0
        for train_index, test_index in cv:
            ## get the data of this cv iteration
            train, test = exputil.sample_data(self.data, train_index, test_index)

            ## get the expert and student
            learner = exputil.get_learner(cfgutil.get_section_options(self.config, 'learner'),
                                          vct=self.vct, sent_tk=self.sent_tokenizer, seed=(t*10+10))

            expert = exputil.get_expert(cfgutil.get_section_options(self.config, 'expert'))

            expert.fit(train.data, y=train.target, vct=self.vct)

            ## do active learning
            results = self.main_loop(learner, expert, self.budget, self.bootstrap_size, train, test)

            ## save the results
            trial.append(results)
            t += 1
        self.report_results(trial)

    def get_name(self):
        cfg = cfgutil.get_section_options(self.config, 'learner')

        name = "data-{}-lrn-{}-ut-{}-snip-{}-cal-{}".format(self.dataname,cfg['type'], cfg['utility'],cfg['snippet'],cfg['calibrate'])
        return name

    def bootstrap(self, pool, bt, train):
        # get a bootstrap
        bt_obj = BootstrapFromEach(None, seed=self.seed)
        initial = bt_obj.bootstrap(pool, step=bt, shuffle=False)

       # update initial training data
        train.index = initial
        train.target = pool.target[initial].tolist()
        return train

    def update_cost(self, current_cost, query):
        return current_cost + self.costfn(query.bow)

    def evaluate(self, learner, test):
        prediction = learner.predict(test.bow)
        pred_proba = learner.predict_proba(test.bow)
        accu = metrics.accuracy_score(test.target, prediction)
        auc = metrics.roc_auc_score(test.target, pred_proba[:,1])
        return {'auc': auc, 'accuracy': accu}

    def evaluate_oracle(self, query, predictions, labels=None):
        cm = metrics.confusion_matrix(query.target, predictions, labels=labels)
        return cm

    def update_run_results(self, results, step, oracle, iteration):
        results['accuracy'][iteration].append(step['accuracy'])
        results['auc'][iteration].append(step['auc'])
        results['ora_accu'][iteration].append(oracle)
        if self.verbose:
            if iteration==self.step:
                print "IT\tACCU\tAUC\tT0\tF1\tF0\tT1"
            print "{0}\t{1:.3f}\t{2:.3f}\t".format(iteration, step['accuracy'],step['auc']),
            print "\t".join(["{0:.3f}".format(x) for x in np.reshape(oracle,4)])
        return results

    def update_pool(self, pool, query, labels, train):
        ## remove from remaining
        for q, t in zip(query.index, labels):
            pool.remaining.remove(q)
            train.index.append(q)
            train.target.append(t)

        return pool, train

    def retrain(self, learner, pool, train):
        X = pool.bow[train.index]
        y = train.target
        ## get training document text
        text = pool.data[train.index]

        return learner.fit(X, y, doc_text=text)

    def main_loop(self, learner, expert, budget, bootstrap, pool, test):
        from  collections import deque

        iteration = 0
        current_cost = 0
        rnd_set = range(pool.target.shape[0])
        self.rnd_state.shuffle(rnd_set)
        remaining = deque(rnd_set)
        pool.remaining = remaining

        ## record keeping
        results = self._start_results()

        ## keep track of current training
        train = bunch.Bunch(index=[], target=[])

        while current_cost <= budget and iteration <= self.max_iteration and pool.remaining > self.step:
            if iteration == 0:
                # bootstrap
                train = self.bootstrap(pool, bootstrap, train)
                learner = self.retrain(learner, pool, train)
            else:
                # select query and query labels
                query = learner.next(pool, self.step)
                labels = expert.label(query.bow, y=query.target)

                # update pool and cost
                pool, train = self.update_pool(pool, query, labels, train)
                current_cost = self.update_cost(current_cost, query)

                # re-train the learner
                learner = self.retrain(learner, pool, train)

                # evaluate
                step_results = self.evaluate(learner, test)
                step_oracle = self.evaluate_oracle(query, labels, labels=np.unique(pool.target))

                # record results
                results = self.update_run_results(results, step_results, step_oracle, current_cost)
                if self.debug:
                    self._debug(learner, expert, query)
            iteration += 1
        return results

    def _start_results(self):
        r = {}
        r['accuracy'] = defaultdict(lambda: [])
        r['auc'] = defaultdict(lambda: [])
        r['ora_accu'] = defaultdict(lambda: [])
        return r

    def _get_iteration(self, iteration):
        cost = sorted(iteration.keys())
        perf = [np.mean(iteration[xi]) for xi in cost]
        std = [np.std(iteration[xi]) for xi in cost]
        n = [np.size(iteration[xi]) for xi in cost]

        return cost, perf, std, n

    def _get_cm_iteration(self, iteration):
        cost = sorted(iteration.keys())
        perf = [np.mean(iteration[xi],axis=0).reshape(4) for xi in cost]
        std = [np.std(iteration[xi]) for xi in cost]
        n = [np.size(iteration[xi]) for xi in cost]

        return cost, perf, std, n

    def report_results(self, results):
        output_name = self.output + "/" + self.get_name()
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        accu = []
        auc = []
        ora = []
        cost = []
        for tr in results:
            c, p, s, n = self._get_iteration(tr['accuracy'])
            accu.append(p)
            c, p, s, n = self._get_iteration(tr['auc'])
            auc.append(p)
            cost.append(c)
            c, p, s, n = self._get_cm_iteration(tr['ora_accu'])
            ora.append(p)

        p = np.mean(accu, axis=0)
        c = np.mean(cost, axis=0)
        s = np.std(accu, axis=0)
        exputil.print_file(c,p,s,open(output_name+"-accu.txt", "w"))
        p = np.mean(auc, axis=0)
        s = np.std(auc, axis=0)
        exputil.print_file(c,p,s,open(output_name+"-auc.txt", "w"))
        p = np.mean(ora, axis=0)
        s = np.std(ora, axis=0)
        exputil.print_cm_file(c,p,s,open(output_name+"-oracle-cm.txt", "w"))

    def _debug(self, learner, expert, query):
        st_prob = learner.snippet_model.predict_proba(query.bow)
        ex_prob = expert.oracle.predict_proba(query.bow)
        for i in len(query.index):
            print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(query.index[i], query.text[i], query.target[i],
                                                      st_prob[i][0], st_prob[i][1], ex_prob[i][0], ex_prob[i][1])