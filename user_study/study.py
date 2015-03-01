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
        self.output ='./results/'

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
        self.learner1 = None
        self.learner2 = None

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

            names = ", ".join(["{}={}".format(a, b) for a, b in enumerate(target_names + ['neutral'])]) + " ? > "
            expert = HumanExpert(None, names)
        else:
            raise Exception("Oops, cannot handle an %s expert" % type_exp)

        return expert

    def get_student(self, config, pool, sequence):

        l1 = cfgutil.get_section_options(config, 'learner1')

        student1 = exputil.get_learner(l1, vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)

        self.learner1 = bunch.Bunch(student=student1, name="{}-{}".format(l1['utility'], l1['snippet']),
                                    pool=pool, train=[], budget=0, sequence=sequence)

        l1 = cfgutil.get_section_options(config, 'learner2')

        student2 = exputil.get_learner(l1, vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)

        self.learner2 = bunch.Bunch(student=student2, name="{}-{}".format(l1['utility'], l1['snippet']),
                                    pool=pool, train=[], budget=0, sequence=sequence)
        return self.learner1, self.learner2

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
        expert_labels['true'].append(query.target)
        expert_labels['labels'].extend(labels)
        expert_labels['time'].append(time)
        expert_labels['data'].append(query.text)
        expert_labels['snip'].append(query.snippet)
        return expert_labels

    def start_record(self):
        r = {}
        r['index'] = []
        r['labels'] = []
        r['true'] = []
        r['time'] = []
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

    def save_results(self, students, expert_times, expert_labels):
        '''
        Save student labels obtained from study and oracle measures
        :return:
        '''

        output_name = self.output + "/" + students['learner1'].name
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        self._save_student_labels(students['learner1'], filename=output_name + "-student-labels.txt")

        output_name = self.output + "/" + students['learner2'].name
        self._save_student_labels(students['learner2'], filename=output_name + "-student-labels.txt")

        self._save_oracle_labels(expert_labels)

    def _save_student_labels(self, student, filename='student'):
        f = open(filename, "w")
        f.write("doc_index\ttrue_label\texp_label")
        for i, doc_i in enumerate(student.train.index):
            f.write("{}\t{}\t{}\n".format(doc_i, student.pool.target[doc_i], student.train.target[i]))
        f.close()

    def _save_oracle_labels(self, expert, filename='expert'):

        f = open(filename, "w")
        f.write("doc_index\ttrue_label\texp_label\tsnippet\tdoc_text")
        n = len(expert['index'])
        for i in range(n):
            txt = expert['text'][i].replace('\n',' ').replace('\t',' ')
            snip = expert['snippet'][i].replace('\n',' ').replace('\t',' ')
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(expert['index'][i], expert['true'][i], expert['labels'][i],
                                                      expert['time'][i], snip, txt))
        f.close()


    def set_options(self, config_obj):
        self.rnd_state = np.random.RandomState(32564)

        config = cfgutil.get_section_options(config_obj, 'data')
        self.data_cat = config['categories']
        self.data_path = config['path']
        self.split = config['split']
        self.vct = exputil.get_vectorizer(config)

        config = cfgutil.get_section_options(config_obj, 'expert')
        self.sent_tokenizer = exputil.get_tokenizer(config['sent_tokenizer'])

        config = cfgutil.get_section_options(config_obj, 'experiment')
        self.seed = config['seed']
        self.budget = config['budget']
        self.step = config['stepsize']
        self.output = config['outputdir']

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
        return current_cost + query.data.shape[0]


    def evaluate_oracle(self, query, predictions, labels=None):
        t = np.array([[x, y] for x, y in zip(query.target, predictions) if y is not None])
        cm = np.zeros((2, 2))
        if len(t) > 0:
            cm = metrics.confusion_matrix(t[:, 0], t[:, 1], labels=labels)
        return cm

    def _sample_data(self, data, train_idx, test_idx):
        sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch())

        if len(test_idx) > 0:  # if there are test indexes
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

        sequence = self.get_sequence(len(self.data.train.target))
        pool, test = self._sample_data(self.data, sequence, [])
        student1, student2 = self.get_student(self.config, pool, sequence)

        expert = self.get_expert(self.config, self.data.train.target_names)

        combined_budget = self.budget * 2
        coin = np.random.RandomState(9187465)

        train = bunch.Bunch(index=[], target=[])
        i = 0
        # expert_labels = self.start_record()
        student = {'learner1':student1, 'learner2':student2}
        expert_times = {'learner1':[], 'learner2':[]}
        expert_labels = {'learner1': self.start_record(), 'learner2': self.start_record()}

        while combined_budget < (2 * self.budget):
            if i == 0:
                ## Bootstrap
                # bootstrap
                train = self.bootstrap(pool, 50, train)
                student['learner1'].student = self.retrain(student['learner1'].student, student['learner1'].pool,
                                                           student['learner1'].train)
                student['learner2'].student = self.retrain(student['learner2'].student, student['learner2'].pool,
                                                           student['learner2'].train)
            else:
                # select student
                next_turn = coin.random_sample()
                print next_turn

                if next_turn < .5:
                    curr_student = 'leaner1'
                else:  # first1 student
                    curr_student = 'learner2'

                query, labels = self.al_cycle(student, expert)

                if query is not None and labels is not None:
                    # re-train the learner
                    student[curr_student].student = self.retrain(student[curr_student].student,
                                                                 student[curr_student].pool, student[curr_student].train)

                    #We can evaluate later
                    step_oracle = self.evaluate_oracle(query, labels, labels=np.unique(student[curr_student].pool.target))

                    # record labels
                    expert_labels[curr_student] = self.record_labels(expert_labels[curr_student], query, labels,
                                                                     time=expert.get_annotation_time())

                    if self.debug:
                        self._debug(student[curr_student], expert, query, step_oracle)

                    combined_budget = student['learner1'].budget + student['learner2'].budget

            i += 1

        self.save_results(student, expert_times, expert_labels)
        ##TODO evaluate the students after getting labels
        self.evaluate_student(student['learner1'], sequence, self.data)
        self.evaluate_student(student['learner2'], sequence, self.data)

    def al_cycle(self, student, expert):
        query = None
        labels = None
        if student.budget <= self.budget:
            query = student.student.next(student.pool, self.step)
            labels = expert.label(query.snippet, y=query.target)

            # update pool and cost
            student.pool, student.train = self.update_pool(student.pool, query, labels, student.train)
            student.budget = self.update_cost(student.budget, query)

        return query, labels


    def _debug(self, student, expert, query, step_oracle):
        print "Student: %s" % student
        print "Time: %.2f" % expert.get_annotation_time()
        print "Oracle CM"
        print "\n".join(["{}\t{}".format(*r) for r in step_oracle])