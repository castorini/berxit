import os
import numpy as np

formal_name = {
    "shrink-1": "weight-tok",
    "all": "joint",
    "all_alternate": "alt",
    "all_alternate-Qvlstm": "alt-Q",
    "all_alternate-lte": "alt-lte",
    "two_stage": "2stg",
    "self_distil": "sd",
}


def np_load(fname):
    return np.load(fname, allow_pickle=True)


def get_etp(x):
    kw = 'entropy_'
    try:
        return float(x[x.index(kw)+len(kw):x.index('.npy')])
    except ValueError:
        return None


class Data:

    def __init__(self, model, dataset, routine, testset=False):
        self.model = model
        self.size = 12 if model.endswith('base') else 24
        self.dataset = dataset
        self.routine = routine
        self.formal_routine = routine
        if routine in formal_name:
            self.formal_routine = formal_name[routine]
        self.filepath = f"saved_models/{self.model}/{self.dataset}/{self.routine}-42/"
        if routine.endswith('lte'):
            self.etp_data = self.get_lte_data(testset=testset)
        else:
            self.layer_acc = self.get_layer_acc()
            self.etp_data = self.get_etp_data(testset=testset)
        self.etp_acc = [[], []]  # x and y for plotting
        for etp, data in self.etp_data:
            self.etp_acc[0].append(data['mean_exit'])
            self.etp_acc[1].append(data['acc'])

    @staticmethod
    def get_mean_exit(counter):
        nom, den, layers = 0, 0, len(counter)
        for i, c in counter.items():
            nom += i * c
            den += c
        return nom / den

    def get_etp_data(self, testset=False):
        filepath = self.filepath + ('/testset/' if testset else '')
        nextlevel_fnames = os.listdir(filepath)
        col = []
        for fname in nextlevel_fnames:
            etp = get_etp(fname)
            if etp is not None:
                np_data = np_load(filepath+fname)
                col.append([etp, {
                    'layer_counter': np_data[0],
                    'time': np_data[1],
                    'mean_exit': self.get_mean_exit(np_data[0]),
                    'acc': np_data[3],
                }])
        col.sort(key=lambda x: x[1]['mean_exit'])
        return col

    def get_layer_acc(self):
        data = np_load(self.filepath + 'each_layer.npy')
        # if self.dataset=='MNLI' and self.model=='bert-base' and self.routine=='limit':
        #     return np.array([x for i, x in enumerate(data)
        #                      if i%2==1])  # MNLI-mm only
        # else:
        return data

    def get_lte_data(self, testset=False):
        if not testset:
            data = np_load(self.filepath + 'lte.npy')
        else:
            data = np_load(self.filepath + 'lte-test.npy')
        col = []
        for entry in data:
            col.append([entry[4], {
                'layer_counter': entry[0],
                'time': entry[1],
                'mean_exit': self.get_mean_exit(entry[0]),
                'acc': entry[3],
            }])
        col.sort(key=lambda x: x[1]['mean_exit'])
        return col


class DistilbertData:
    def __init__(self):
        self.saving = 410/668
        self.acc = {
            'RTE': 59.9,
            'MRPC': 87.5,
            'SST-2': 91.3,
            'QNLI': 89.2,
            'QQP': 88.5,  # accuracy not f1
            'MNLI': 82.2,
            'STS-B': 86.9,
        }

class RawBertData:
    def __init__(self, size):
        if size=='base':
            self.layers = 12
            self.acc = {
                'RTE': 66.4,
                'MRPC': 88.9,
                'SST-2': 93.5,
                'QNLI': 90.5,
                'QQP': 71.2,
                'MNLI': 83.4,
                'STS-B': 85.8,
            }
        elif size=='large':
            self.layers = 24
            self.acc = {
                'RTE': 70.1,
                'MRPC': 89.3,
                'SST-2': 94.9,
                'QNLI': 92.7,
                'QQP': 72.1,
                'MNLI': 85.9,
                'STS-B': 86.5,
            }

