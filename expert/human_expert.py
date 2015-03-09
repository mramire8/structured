__author__ = 'maru'

from expert.base import BaseExpert

class HumanExpert(BaseExpert):

    def __init__(self, model, prompt):
        super(HumanExpert, self).__init__(model)
        self.elapsed_time = -1
        self.num_classes = 3  # binary + neutral
        self.prompt = prompt

    def label(self, data, y=None):
        import time
        import textwrap
        labels = []
        times = []
        for doc in data:
            print_text = doc.strip()
            print
            print ("-"*40)
            print
            print '\033[94m' + textwrap.fill(print_text, width=70) +'\033[0m'
            t0 = time.time()
            valid = False
            answer = -1
            while not valid:
                print
                answer = raw_input('\033[92m'+self.prompt+'\033[0m')
                try:
                    answer = int(answer)
                    if answer not in range(0, self.num_classes):
                        valid = False
                    else:
                        valid = True
                        labels.append(answer)
                except ValueError:
                   valid = False
            self.elapsed_time = time.time() - t0
            times.append(self.elapsed_time)
        return labels

    def fit(self, data, y=None):
        pass

    def get_annotation_time(self):
        return self.elapsed_time