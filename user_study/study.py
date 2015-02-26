__author__ = 'maru'

import os, sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

import numpy as np
import utilities.datautils as datautil
import utilities.configutils as cfgutil
import utilities.experimentutils as exputil

from sklearn.datasets import base as bunch
from learner.strategy import BootstrapFromEach


class Study(object):
    def __init__(self, dataname, config, verbose=False, debug=False):
        super(Study, self).__init__()
        self.seed = None
        self.rnd_state = np.random.RandomState(32564)
        self.config = config

        self.dataname = dataname
        self.data_cat = None
        self.data = None
        self.data_path = None
        self.split = None

        self.vct = None
        self.sent_tokenizer = None

        self.budget = None
        self.step = 1
        self.debug = debug
        self.verbose = verbose

    def get_sequence(self, n):
        ''' Get a sequence of document for the study
        :return:
        '''
        seq = self.rnd_state.permutation(n)
        return seq

    def get_expert(self, config, target_names):
        ''' Get human expert
        :return:
        '''
        type_exp = cfgutil.get_section_options(config, 'expert')
        if type_exp == 'human':
            from expert.human_expert import HumanExpert
            names = ", ".join(["{}={}".format(a,b) for a,b in enumerate(target_names + ['neutral'])])+" ? > "
            expert = HumanExpert(None, names)
        else:
            raise Exception("Oops, cannot handle an %s expert" % type_exp)

        return expert

    def get_student(self, config):

        student1 = exputil.get_learner(cfgutil.get_section_options(config, 'learner1'),
                                      vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)
        student2 = exputil.get_learner(cfgutil.get_section_options(config, 'learner2'),
                                      vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)
        return student1, student2

    def evaluate_student(self):
        '''
        After getting labels, train and evaluated students for plotting
        :return:
        '''

        pass

    def record_labels(self):
        '''
        Save data labels from the expert
        :return:
        '''
        pass

    def retrain(self, learner, pool, train):
        '''
        Retrain student for active learning loop
        :return:
        '''
        x = pool.bow[train.index]
        y = train.target
        ## get training document text
        text = pool.data[train.index]

        return learner.fit(x, y, doc_text=text)

    def save_results(self):
        '''
        Save student performances and oracle measures
        :return:
        '''
        pass

    def set_options(self, config):
        pass

    def update_pool(self, pool, query, labels, train):
        ## remove from remaining
        for q, t in zip(query.index, labels):
            pool.remaining.remove(q)
            if t is not None:  # if the answer is not neutral
                train.index.append(q)
                train.target.append(t)

        return pool, train

    def bootstrap(self, pool, bt, train):
        # get a bootstrap
        bt_obj = BootstrapFromEach(None, seed=self.seed)
        initial = bt_obj.bootstrap(pool, step=bt, shuffle=False)

        # update initial training data
        train.index = initial
        train.target = pool.target[initial].tolist()
        return train

    def update_cost(self, current_cost, query):
        return current_cost + query.bow.shape[0]


    def start(self):
        from collections import deque

        self.set_options(self.config)
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed,
                                          shuffle=True, percent=self.split, keep_subject=True)
        student1, student2 = self.get_student(self.config)
        sequence = self.get_sequence(self.budget)

        expert = self.get_expert(self.config, self.data.train.target_names)

        combined_budget = self.budget * 2
        coin = np.random.RandomState(9182837465)

        pool = None
        train = bunch.Bunch(index=[], target=[])
        i = 0
        while combined_budget > 0:
            if i == 0:
                ## Bootstrap
                # bootstrap
                train = self.bootstrap(pool, 50, train)
                student1 = self.retrain(student1, pool, train)
                # student2 = self.retrain(student2, pool, train)
            else:
                #select student
                next = coin.random_sample()
                student = None

                if next < .5:
                    student = student1
                else:
                    student = student2

                # select query and query labels
                query = student.next(pool, self.step)
                labels = expert.label(query.data, y=query.target)

                # update pool and cost
                pool, train = self.update_pool(pool, query, labels, train)
                combined_budget = self.update_cost(combined_budget, expert)

                # re-train the learner
                student = self.retrain(student, pool, train)

                # record labels
                self.record_labels(query, labels)
                # step_results = self.evaluate(student, test)
                # step_oracle = self.evaluate_oracle(query, labels, labels=np.unique(pool.target))

                if self.debug:
                    self._debug(student, expert, query)

                # get document in sequence

                #select questies
                #ask queries
                #record answers
                # get codt

            i += 1

        self.evaluate_student(self.student1, sequence, self.data)
        self.evaluate_student(self.student2, sequence, self.data)
