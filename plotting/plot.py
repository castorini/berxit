import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
from get_data import Data

model = sys.argv[1]

formal_name = {
    "shrink-1": "weight-tok",
    "all": "joint",
    "all_alternate": "alternating"
}

color_pool = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {
    "two_stage": color_pool[0],
    "joint": color_pool[1],
    "alternating": color_pool[2],
    "alternating-Q": color_pool[2],
    "weight-tok": color_pool[3],
    "alternate-1": color_pool[4],
    "limit": color_pool[5]
}

datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
sizes = ["2.5k", "3.5k", "67k", "108k", "363k", "392k"]
routines = ["all_alternate", "all_alternate-Qvlstm"]
# routines = ["two_stage", "all", "all_alternate", "limit"]

columns = 3  # 3 for landscape, 2 for portrait
M, N = len(datasets)//columns, columns
fig, axes = plt.subplots(M, N, figsize=[N*4, M*4])
axes.reshape([-1])


def plot_acc_by_layer(axis, data):
    xrange = range(1, 1+data.size)
    axis.set_xlim(1, data.size)

    if data.routine.endswith('Qvlstm'):
        # Qmodule acc
        axis.plot(*data.etp_acc, 'o:', color=colors[data.formal_routine],
                  linewidth=1, markersize=2, label=data.formal_routine)
    else:
        # layer-wise acc
        axis.plot(xrange, data.layer_acc, 'o-', color=colors[data.formal_routine],
                  label=data.formal_routine, linewidth=1, markersize=2)
        # entropy-based acc
        axis.plot(*data.etp_acc, 'o-.', color=colors[data.formal_routine],
                  linewidth=1, markersize=2)


for i_dataset, dataset in enumerate(datasets):
    dataset_axis = axes[i_dataset // columns][i_dataset % columns]
    for i_routine, routine in enumerate(routines):
        try:
            data_obj = Data(model, dataset, routine)
            plot_acc_by_layer(
                dataset_axis,
                data_obj,
            )
        except FileNotFoundError:
            pass
    dataset_axis.set_xlim(0, data_obj.size+1)
    dataset_axis.legend(loc='lower right', fontsize=15)
    dataset_axis.set_title(datasets[i_dataset]+' ('+sizes[i_dataset]+')', fontsize=15)
    dataset_axis.set_xlabel("Exit layer")
    dataset_axis.set_ylabel("Score")




plt.tight_layout()
plt.show()
# plt.savefig(f"{model}.pdf")
