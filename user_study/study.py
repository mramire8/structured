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
from sklearn import metrics

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
        if type_exp['type'] == 'human':
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

    def evaluate_student(self, student, sequence, data):
        '''
        After getting labels, train and evaluated students for plotting
        :return:
        '''

        pass

    def record_labels(self, expert_labels, query, labels, time=None):
        '''
        Save data labels from the expert
        :return:
        '''
        expert_labels['index'].append(query.index)
        expert_labels['labels'].extend(labels)
        expert_labels['time'].extend(time)
        return expert_labels

    def start_record(self):
        r = {}
        r['index'] = []
        r['labels'] = []
        return r

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

    def set_options(self, config_obj):
        self.seed = None
        self.rnd_state = np.random.RandomState(32564)


        config = cfgutil.get_section_options(config_obj, 'data')
        self.data_cat = config['categories']
        self.data_path = config['path']
        self.split = config['split']
        self.vct = exputil.get_vectorizer(config)

        config = cfgutil.get_section_options(config_obj, 'expert')
        self.sent_tokenizer = exputil.get_tokenizer(config['sent_tokenizer'])

        config = cfgutil.get_section_options(config_obj, 'experiment')

        self.budget = config['budget']
        self.step = config['stepsize']


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


    def evaluate_oracle(self, query, predictions, labels=None):
        t = np.array([[x,y] for x,y in zip(query.target, predictions) if y is not None])
        cm = np.zeros((2,2))
        if len(t)> 0:
            cm = metrics.confusion_matrix(t[:,0], t[:,1], labels=labels)
        return cm

    def _sample_data(self, data, train_idx, test_idx):
        sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch())

        if len(test_idx) > 0: #if there are test indexes
            sample.train.data = np.array(data.data, dtype=object)[train_idx]
            sample.test.data = np.array(data.data, dtype=object)[test_idx]

            sample.train.target = data.target[train_idx]
            sample.test.target = data.target[test_idx]

            sample.train.bow = self.vct.fit_transform(sample.train.data)
            sample.test.bow = self.vct.transform(sample.test.data)

            sample.target_names = data.target_names
            sample.train.remaining = []
        else:
            ## Just shuffle the data and vectorize
            sample = data
            data_lst = np.array(data.train.data, dtype=object)
            data_lst = data_lst[train_idx]
            sample.train.data = data_lst

            sample.train.target = data.train.target[train_idx]

            sample.train.bow = self.vct.fit_transform(data.train.data)
            sample.test.bow = self.vct.transform(data.test.data)

            sample.train.remaining = []

        return sample.train, sample.test

    def start(self):
        from collections import deque

        self.set_options(self.config)
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed,
                                          shuffle=True, percent=self.split, keep_subject=True)
        student1, student2 = self.get_student(self.config)
        sequence = self.get_sequence(len(self.data.train.target))

        expert = self.get_expert(self.config, self.data.train.target_names)

        combined_budget = self.budget * 2
        coin = np.random.RandomState(9187465)

        pool, test = self._sample_data(self.data, sequence, [])
        remaining = deque(sequence)
        pool.remaining = remaining

        train = bunch.Bunch(index=[], target=[])
        i = 0
        expert_labels = self.start_record()

        while combined_budget > 0:
            if i == 0:
                ## Bootstrap
                # bootstrap
                train = self.bootstrap(pool, 50, train)
                student1 = self.retrain(student1, pool, train)
                # student2 = self.retrain(student2, pool, train)
            else:
                #select student
                next_turn = coin.random_sample()

                if next_turn < .5:
                    student = student1
                else: # first1 student
                    student = student2

                # select query and query labels
                query = student.next(pool, self.step)
                labels = expert.label(query.snippet, y=query.target)

                # update pool and cost
                pool, train = self.update_pool(pool, query, labels, train)
                combined_budget = self.update_cost(combined_budget, expert)

                # re-train the learner
                if next_turn < .5:
                    student1 = self.retrain(student1, pool, train)
                else:
                    # first1 student no need to train
                    student2 = self.retrain(student2, pool, train)

                # record labels
                expert_labels = self.record_labels(expert_labels, query, labels, time=expert.get_annotation_time())
                # step_results = self.evaluate(student, test)

                #We can evaluate later
                # step_oracle = self.evaluate_oracle(query, labels, labels=np.unique(pool.target))
                print self.evaluate_oracle(query, labels, labels=np.unique(pool.target))


                if self.debug:
                    #TODO print out information?
                    # self._debug(student, expert, query)
                    pass

            i += 1

        ##TODO evaluate the students after getting labels
        self.evaluate_student(student1, sequence, self.data)
        self.evaluate_student(student2, sequence, self.data)
