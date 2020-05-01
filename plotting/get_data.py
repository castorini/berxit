import os
import numpy as np

formal_name = {
    "shrink-1": "weight-tok",
    "all": "joint",
    "all_alternate": "alternating"
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

    def __init__(self, model, dataset, routine):
        self.model = model
        self.dataset = dataset
        self.routine = routine
        self.formal_routine = routine
        if routine in formal_name:
            self.formal_routine = formal_name[routine]
        self.filepath = f"saved_models/{self.model}/{self.dataset}/{self.routine}-42/"
        self.etp_data = self.get_etp_data()
        if len(self.etp_data) == 0:
            self.not_finished = True
        else:
            self.not_finished = False
        self.layer_acc = self.get_layer_acc()
        self.etp_acc = [[], []]  # x and y for plotting
        for etp, data in self.etp_data:
            self.etp_acc[0].append(data['mean_exit'])
            self.etp_acc[1].append(data['acc'])

    def get_etp_data(self):
        def get_mean_exit(counter):
            nom, den, layers = 0, 0, len(counter)
            for i, c in counter.items():
                nom += i*c
                den += c
            return nom/den

        nextlevel_fnames = os.listdir(self.filepath)
        col = []
        for fname in nextlevel_fnames:
            etp = get_etp(fname)
            if etp is not None:
                np_data = np_load(self.filepath+fname)
                col.append([etp, {
                    'layer_counter': np_data[0],
                    'time': np_data[1],
                    'mean_exit': get_mean_exit(np_data[0]),
                    'acc': np_data[3],
                }])
        col.sort(key=lambda x: x[0])
        return col

    def get_layer_acc(self):
        data = np_load(self.filepath + 'each_layer.npy')
        if self.dataset=='MNLI' and self.model=='bert-base' and self.routine=='limit':
            return np.array([x for i, x in enumerate(data)
                             if i%2==1])  # MNLI-mm only
        else:
            return data
