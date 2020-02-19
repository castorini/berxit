import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns

model = sys.argv[1]

def np_load(fname):
    return np.load(fname, allow_pickle=True)

datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
sizes = ["2.5k", "3.5k", "67k", "108k", "363k", "392k"]
routines = ["two_stage", "all", "shrink-1", "self_distil"]
M, N = 3, len(datasets)//3
fig, axes = plt.subplots(M, N, figsize=[N*4, M*4])
axes.reshape([-1])

def plot_acc_by_layer(axis, datafile, title):
    # M = [max(d[0]) for d in data if d[1]=='two_stage'][0]
    for d in data:
        # d[0] = [x/M for x in d[0]]
        if d[1]=='shrink-1':
            d[1] = 'weight-tok'
        axis.plot(d[0], 'o-', label=d[1],
                  linewidth=1, markersize=1)
    axis.legend(loc='lower right', fontsize=15)
    axis.set_title(title, fontsize=15)
    axis.set_xlabel("Exit layer")
    axis.set_ylabel("Score")




for i in range(len(datasets)):
    data = []
    for j in range(len(routines)):
        try:
            data.append(
                [np_load(f"saved_models/{model}/{datasets[i]}/{routines[j]}-42/each_layer.npy"),
                 routines[j]]
            )
        except Exception as e:
            print(e)
            pass
    plot_acc_by_layer(
        axes[i//N][i%N],
        data,
        title=datasets[i]+' ('+sizes[i]+')'
    )
plt.tight_layout()
# plt.show()
plt.savefig(f"{model}.pdf")