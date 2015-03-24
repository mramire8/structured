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
        self.bootstrap_size = None
        self.step = 1
        self.debug = debug
        self.verbose = verbose
        self.learner1 = None
        self.learner2 = None
        self.prefix = ""

    def get_sequence(self, n, subsample):
        ''' Get a sequence of document for the study
        :return:
        '''
        # seq = self.rnd_state.choice(range(n), subsample, False)
        seq = self.rnd_state.permutation(n)

        return seq

    def get_expert(self, config, target_names):
        ''' Get human expert
        :return:
        '''
        type_exp = cfgutil.get_section_options(config, 'expert')
        if type_exp['type'] == 'human':
            from expert.human_expert import HumanExpert

            names = ", ".join(["{}={}".format(a, b) for a, b in enumerate(target_names + ['neutral'])]) \
                    + " pause=p" + " > "
            expert = HumanExpert(None, names)
        else:
            raise Exception("Oops, cannot handle an %s expert" % type_exp)

        return expert

    def get_student(self, config, pool, sequence):
        from collections import deque
        l1 = cfgutil.get_section_options(config, 'learner1')

        pool[0].remaining = deque(sequence)
        student1 = exputil.get_learner(l1, vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)

        self.learner1 = bunch.Bunch(student=student1, name="{}-{}".format(l1['utility'], l1['snippet']),
                                    pool=pool[0], train=[], budget=0, sequence=sequence)

        l1 = cfgutil.get_section_options(config, 'learner2')

        student2 = exputil.get_learner(l1, vct=self.vct, sent_tk=self.sent_tokenizer, seed=self.seed)

        ## reshuffle the sequence
        rnd2 = np.random.RandomState(9187465)
        sequence2 = [s for s in sequence]
        rnd2.shuffle(sequence2)

        # udpade the pool
        pool[1].remaining = deque(sequence2)
        self.learner2 = bunch.Bunch(student=student2, name="{}-{}".format(l1['utility'], l1['snippet']),
                                    pool=pool[1], train=[], budget=0, sequence=sequence2)
        return self.learner1, self.learner2

    def _evaluate(self, learner, test):
        prediction = learner.predict(test.bow)
        pred_proba = learner.predict_proba(test.bow)
        accu = metrics.accuracy_score(test.target, prediction)
        auc = metrics.roc_auc_score(test.target, pred_proba[:, 1])
        return {'auc': auc, 'accuracy': accu}


    def evaluate_student(self, student_clf, train, sequence, data, test, name='', order=False):
        '''
        After getting labels, train and evaluated students for plotting
        :return:
        '''
        from collections import defaultdict

        if order:
            # train = student.train
            # order1 = np.argsort(sequence[self.bootstrap_size:])
            # order2 = np.argsort(train.index)
            # new_target = np.array(train.target)[order1][order2]
            new_target = [train.target[train.index.index(i)] for i in sequence[self.bootstrap_size:]]
            train.target = np.array(new_target, dtype=int)
            train.index = sequence

        x = range(self.bootstrap_size, len(sequence)+1, 1)
        cost = np.array(x, dtype=object)
        results = dict()
        results['accuracy'] = defaultdict(lambda: [])
        results['auc'] = defaultdict(lambda: [])
        results['ora_accu'] = defaultdict(lambda: [])

        for i, c in enumerate(cost):
            if i == 0:
                train_idx = sequence[:c]
                train_target = data.target[train_idx]

            else:
                # train student
                non_neutral = np.array(train.target[:i]) < 2
                train_idx = np.append(np.array(sequence[:self.bootstrap_size]), np.array(train.index[:i])[non_neutral])
                train_target = np.append(data.target[sequence[:self.bootstrap_size]], np.array(train.target[:i])[non_neutral])

            # print np.unique(train_target),  len(train_target)

            student_clf.fit(data.bow[train_idx], train_target)

            # test
            prediction = student_clf.predict(test.bow)
            pred_proba = student_clf.predict_proba(test.bow)

            accu = metrics.accuracy_score(test.target, prediction)
            auc = metrics.roc_auc_score(test.target, pred_proba[:, 1])

            oracle = metrics.confusion_matrix(data.target[train_idx], train_target, labels=[0,1])
            # record
            results['accuracy'][c].append(accu)
            results['auc'][c].append(auc)
            results['ora_accu'][c].append(oracle)

            print c, accu, auc

        #save all
        self.save_evaluation_results(results, name=name)

    def save_evaluation_results(self, results, name='student'):
        output_name = self.output + "/" + self.dataname + "-" + name + "-curve" + self.prefix + "-"

        accu = results['accuracy']
        xaxis = results['accuracy'].keys()
        y = [np.mean(accu[x]) for x in xaxis]

        self._print_file(xaxis, y, output_name + "accu.txt")

        accu = results['auc']
        xaxis = results['auc'].keys()
        y = [np.mean(accu[x]) for x in xaxis]

        self._print_file(xaxis, y, output_name + "auc.txt")

        accu = results['ora_accu']
        xaxis = results['ora_accu'].keys()
        y = ["\t".join("{}\t{}".format(*xx) for xx in accu[x]) for x in xaxis]

        self._print_file2(xaxis, y, output_name + "cm.txt")

    def record_labels(self, expert_labels, query, labels, time=None, pause=None):
        '''
        Save data labels from the expert
        :return:
        '''
        expert_labels['index'].extend(query.index)
        expert_labels['true'].extend(query.target)
        expert_labels['labels'].extend(labels)
        expert_labels['time'].append(time)
        expert_labels['text'].extend(query.text)
        expert_labels['snip'].extend(query.snippet)
        expert_labels['paused'].append(pause)
        return expert_labels

    def start_record(self):
        r = {}
        r['index'] = []
        r['labels'] = []
        r['true'] = []
        r['time'] = []
        r['text'] = []
        r['snip'] = []
        r['paused'] = []
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

        output_name = self.output + "/" + self.dataname+"-"+students['learner1'].name
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        self._save_student_labels(students['learner1'], filename=output_name + "-student-labels.txt")

        output_name = self.output + "/" + self.dataname+"-"+students['learner2'].name
        self._save_student_labels(students['learner2'], filename=output_name + "-student-labels.txt")

        output_name = self.output + "/" + self.dataname+"-"+students['learner1'].name
        self._save_oracle_labels(expert_labels['learner1'], filename=output_name + "-expert1-labels.txt")

        output_name = self.output + "/" + self.dataname+"-"+ students['learner2'].name
        self._save_oracle_labels(expert_labels['learner2'], filename=output_name + "-expert2-labels.txt")

    def _save_student_labels(self, student, filename='student'):
        f = open(filename, "w")
        f.write("doc_index\ttrue_label\texp_label\tdoc_text\n")
        for i, doc_i in enumerate(student.train.index):
            doc_text = student.pool.data[doc_i]
            doc_text = doc_text.encode('latin1').replace('\n',' ').replace('\t',' ')
            f.write("{}\t{}\t{}\t{}\n".format(doc_i, student.pool.target[doc_i], student.train.target[i], doc_text))
        f.close()

    def _save_oracle_labels(self, expert, filename='expert'):

        f = open(filename, "w")
        f.write("doc_index\ttrue_label\texp_label\tannotation_time\tpausedp_time\tsnippet\tdoc_text\n")
        exp = expert
        n = len(exp['index'])
        for i in range(n):
            txt = exp['text'][i].encode('latin1').replace('\n',' ').replace('\t',' ')
            snip = exp['snip'][i].encode('latin1').replace('\n',' ').replace('\t',' ')
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(exp['index'][i], exp['true'][i], exp['labels'][i],
                                                      exp['time'][i], exp['paused'][i], snip, txt))
        f.close()

    def _print_file(self, x, y, file_name):
        import os
        dir_file = os.path.dirname(file_name)
        if not os.path.exists(dir_file):
            os.makedirs(dir_file)

        f = open(file_name, "w")
        f.write("COST\tMEAN\n")
        for a, b in zip(x, y):
            f.write("{0:.3f}\t{1:.3f}\n".format(a, b))
        f.close()

    def _print_file2(self, x, y, file_name):
        import os
        dir_file = os.path.dirname(file_name)
        if not os.path.exists(dir_file):
            os.makedirs(dir_file)

        f = open(file_name, "w")
        f.write("COST\tMEAN\n")
        for a, b in zip(x, y):
            f.write("{}\t{}\n".format(a, b))
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
        self.bootstrap_size = config['bootstrap']
        self.prefix = config['fileprefix']

    def update_pool(self, pool, query, labels, train):
        ## remove from remaining
        for q, t in zip(query.index, labels):
            pool.remaining.remove(q)
            if t is not None and t < 2:  # if the answer is not neutral
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

    def copy_pool(self, obj):
        import copy
        pool2= bunch.Bunch(data=copy.copy(obj.data), bow=copy.copy(obj.bow),
                           target=copy.copy(obj.target), remaining=copy.copy(obj.remaining))
        return pool2

    def start(self):
        import copy
        from collections import deque
        from time import time

        self.set_options(self.config)
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed,
                                          shuffle=True, percent=self.split, keep_subject=True)

        sequence = self.get_sequence(len(self.data.train.target), self.budget+self.bootstrap_size)

        pool, test = self._sample_data(self.data, sequence, [])
        # pool2, _ = self._sample_data(self.data, sequence, [])
        # pool2 = copy.deepcopy(pool)
        pool2 = self.copy_pool(pool)
        # pool2.remaining = []

        student1, student2 = self.get_student(self.config, [pool, pool2], sequence)

        expert = self.get_expert(self.config, self.data.train.target_names)

        combined_budget = 0
        coin = np.random.RandomState(9187465)

        i = 0
        # expert_labels = self.start_record()
        student = {'learner1':student1, 'learner2':student2}
        expert_times = {'learner1':[], 'learner2':[]}
        expert_labels = {'learner1': self.start_record(), 'learner2': self.start_record()}
        original_sequence = []

        raw_input("\n*** Press <return> to start ***")

        t0 = time()
        while combined_budget < (2 * self.budget):
            if i == 0:
                ## Bootstrap
                # bootstrap
                train = self.bootstrap(student['learner1'].pool, self.bootstrap_size, bunch.Bunch(index=[], target=[]))

                student['learner1'].train = train
                student['learner2'].train = bunch.Bunch(index=copy.copy(train.index), target=copy.copy(train.target))

                student['learner1'].student = self.retrain(student['learner1'].student, student['learner1'].pool,
                                                           student['learner1'].train)

                student['learner2'].student = self.retrain(student['learner2'].student, student['learner2'].pool,
                                                           student['learner2'].train)

                for t in train.index:
                    student['learner1'].pool.remaining.remove(t)
                    student['learner2'].pool.remaining.remove(t)

                tmp_list = list(student['learner1'].pool.remaining)
                pool_sample = self.rnd_state.choice(tmp_list, self.budget, False)
                student['learner1'].pool.remaining = deque(pool_sample)
                original_sequence = copy.copy(train.index) + list(pool_sample)

                self.rnd_state.shuffle(pool_sample)
                student['learner2'].pool.remaining = deque(pool_sample)
            else:
                # select student
                next_turn = coin.random_sample()

                if next_turn < .5:
                    curr_student = 'learner1'
                else:  # first1 student
                    curr_student = 'learner2'

                query, labels = self.al_cycle(student[curr_student], expert)

                # print len(student['learner1'].pool.remaining), len(student['learner2'].pool.remaining)

                if query is not None and labels is not None:

                    # progress
                    print "\n%.1f %% completed" % (100. * combined_budget / (2 * self.budget))
                    # re-train the learner
                    student[curr_student].student = self.retrain(student[curr_student].student,
                                                                 student[curr_student].pool, student[curr_student].train)

                    #We can evaluate later
                    step_oracle = self.evaluate_oracle(query, labels, labels=np.unique(student[curr_student].pool.target))

                    # record labels
                    expert_labels[curr_student] = self.record_labels(expert_labels[curr_student], query, labels,
                                                                     time=expert.get_annotation_time(),
                                                                     pause=expert.get_pause())

                    if self.debug:
                        self._debug(student[curr_student], expert, query, step_oracle)

                    combined_budget = student['learner1'].budget + student['learner2'].budget

            i += 1

        t1 = time()
        print "\nTotal annotation time: %.3f secs (%.3f mins)" % ((t1-t0), (t1-t0)/60)

        self.save_results(student, expert_times, expert_labels)
        ##TODO evaluate the students after getting labels
        # self.evaluate_student(student['learner1'], student['learner1'].train.index, pool, test, order=False)
        # self.evaluate_student(student['learner2'], student['learner1'].train.index, pool, test, order=True)

        t = bunch.Bunch(index=expert_labels['learner1']['index'], target=expert_labels['learner1']['labels'])
        self.evaluate_student(student['learner1'].student.model, t, original_sequence, pool, test,
                              name=student['learner1'].name, order=False)

        t = bunch.Bunch(index=expert_labels['learner2']['index'], target=expert_labels['learner2']['labels'])
        self.evaluate_student(student['learner2'].student.model, t, original_sequence, pool, test,
                              name=student['learner2'].name, order=True)

    def al_cycle(self, student, expert):
        query = None
        labels = None
        if student.budget < self.budget:
            query = student.student.next(student.pool, self.step)
            labels = expert.label(query.snippet, y=query.target)

            # update pool and cost
            student.pool, student.train = self.update_pool(student.pool, query, labels, student.train)
            student.budget = self.update_cost(student.budget, query)

        return query, labels


    def _debug(self, student, expert, query, step_oracle):
        print "Student: %s" % student.name
        print "Time: %.2f" % expert.get_annotation_time()
        print "Oracle CM"
        print "\n".join(["{}\t{}".format(*r) for r in step_oracle])