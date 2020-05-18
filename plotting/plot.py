import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from get_data import Data, DistilbertData


# default style
sns.set_style("whitegrid")
matplotlib.rc('font', size=20)
matplotlib.rc('text', usetex=True)


# data preparation
model, plot_target = sys.argv[1], int(sys.argv[2])
model_formal_name = {
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'albert': 'ALBERT',
    'distilbert': 'DistilBERT',
}
plot_target_name = [
    'routine_comp',
    'layer_etp_acc_comp',
    'layer_lte_acc_comp',
]
"""
0: comparison between different training routines
1: comparison between layer_acc and etp_acc
2: comparison between layer_acc and lte
"""

testset = False
if len(sys.argv)>3 and sys.argv[3] == 'testset':
    testset = True

formal_name = {
    "shrink-1": "weight-tok",
    "all": "joint",
    "all_alternate": "alternating"
}

color_pool = plt.rcParams['axes.prop_cycle'].by_key()['color']
darkcolors = {
    "two_stage": color_pool[0],  # blue
    "all": 'darkorange',
    "all_alternate": 'darkgreen',
    "all_alternate-lte": color_pool[1],  # orange
    "limit": 'tab:brown'
}
# lightcolors = {
#     "two_stage": 'cornflowerblue',
#     "all": 'orange',
#     "all_alternate": 'mediumseagreen',
# }

if plot_target in [0, 1]:
    datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
    if not model.startswith('bert'):
        datasets = ['RTE', 'MRPC']
    sizes = ["2.5k", "3.5k", "67k", "108k", "363k", "392k"]
    routines = ["two_stage", "all", "all_alternate", "limit"]
    if plot_target == 1:
        routines = ["all", "all_alternate", "limit"]
    columns = 2
elif plot_target == 2:
    datasets = ['STS-B', 'SST-2']
    sizes = ['5.7k', '67k']
    routines = ["all_alternate", "all_alternate-lte"]
    columns = 2

M, N = len(datasets)//columns, columns
fig, axes = plt.subplots(M, N, figsize=[N*4, M*4])
axes = axes.reshape([-1])


def auc(data):
    xs, ys = data
    area = 0
    for i in range(len(xs)-1):
        area += (xs[i+1] - xs[i]) * (ys[i+1] + ys[i-1]) / 2  # trapezoid
    return area


def plot_acc_by_layer(axis, data):
    axis.set_xlim(1, data.size)

    if data.routine.endswith('lte'):
        # lte acc
        axis.plot(*data.etp_acc, 'o-', color=darkcolors[data.routine],
                  linewidth=1, markersize=2,
                  label='\\textsc{' + data.formal_routine + '}'
                  )
    else:
        # layer-wise acc
        if (
                (plot_target == 0 and data.routine != 'limit') or
                (plot_target == 1 and data.routine == 'limit') or
                (plot_target == 2)
        ):
            axis.plot(range(1, 1+len(data.layer_acc)), data.layer_acc,
                      'o-', color=darkcolors[data.routine],
                      label='\\textsc{' + data.formal_routine + '}',
                      linewidth=1, markersize=2)

        # entropy-based acc
        if plot_target == 1 and data.routine != 'limit':
            axis.plot(*data.etp_acc, 'o-', color=darkcolors[data.routine],
                      label='\\textsc{' + data.formal_routine + '}',
                      linewidth=1, markersize=2)


distilbert_data = DistilbertData()
for i_dataset, dataset in enumerate(datasets):
    dataset_axis = axes[i_dataset]

    if plot_target==1 and model=='bert-base':
        dataset_axis.scatter(
            distilbert_data.saving * 12,
            distilbert_data.acc[dataset] / 100,
            s=100,  # size
            marker='+',
            # label='\\textsc{dist}'
        )

    for i_routine, routine in enumerate(routines):
        try:
            data_obj = Data(model, dataset, routine, testset=testset)
            plot_acc_by_layer(
                dataset_axis,
                data_obj,
            )
        except FileNotFoundError:
            pass

    try:
        dataset_axis.set_xlim(0, data_obj.size+1)

        # to avoid legend overlapping
        if dataset == 'RTE' and model == 'albert-base' and plot_target==0:
            dataset_axis.set_ylim(bottom=0.35)
        if dataset == 'RTE' and model == 'albert-base' and plot_target==1:
            dataset_axis.set_ylim(bottom=0.2)

        dataset_axis.legend(loc='lower right',
                            prop = {'weight': 'bold'},
                            fontsize=15)
        model_name, model_size = model.split('-')
        model_name = model_formal_name[model_name]
        model_size = '\\textsubscript{\\textsc{' + model_size + '}}'
        dataset_axis.set_title(model_name + model_size + ' : ' + datasets[i_dataset])
        dataset_axis.set_xlabel(
            ('(Avg.) ' if plot_target>0 else '') +
            "Exit layer"
        )
        dataset_axis.set_ylabel("Score")
    except NameError:
        pass



plt.tight_layout()
plt.show()
# plt.savefig(f"{model}-{plot_target_name[plot_target]}.pdf")
