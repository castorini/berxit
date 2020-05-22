import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from get_data import Data, DistilbertData, RawBertData


# default style
sns.set_style("whitegrid")
matplotlib.rc('font', size=20)
matplotlib.rc('text', usetex=True)

RELATIVE = True


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

color_pool = plt.rcParams['axes.prop_cycle'].by_key()['color']
darkcolors = {
    "two_stage": color_pool[0],  # blue
    "all": 'darkorange',
    "all_alternate": 'darkgreen',
    "all_alternate-lte": color_pool[1],  # orange
    "self_distil": color_pool[1],
    "limit": 'tab:brown'
}
# lightcolors = {
#     "two_stage": 'cornflowerblue',
#     "all": 'orange',
#     "all_alternate": 'mediumseagreen',
# }

if plot_target in [0, 1]:
    if model.startswith('bert') or model == 'roberta-large':
        datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
    else:
        datasets = ['RTE', 'MRPC']
    sizes = ["2.5k", "3.5k", "67k", "108k", "363k", "392k"]
    routines = ["two_stage", "all", "all_alternate", "limit"]
    if plot_target == 1:
        routines = ["all", "all_alternate", "limit"]
    if model in ['bert-large', 'roberta-large']:
        routines = ['two_stage', 'all_alternate', 'self_distil']
    columns = 2
elif plot_target == 2:
    datasets = ['STS-B', 'SST-2']
    sizes = ['5.7k', '67k']
    routines = ["all_alternate", "all_alternate-lte"]
    columns = 2

rna = False
if model == 'rna-base':  # roberta and albert
    datasets = datasets + datasets
    rna = True

M, N = len(datasets)//columns, columns
if plot_target == 2:
    if model == 'bert-base':
        sizes = [N*4, M*4+0.8]
    else:
        sizes = [N*4, M*4]
else:
    sizes = [N*4, M*4+1]
fig, axes = plt.subplots(M, N, figsize=sizes)
axes = axes.reshape([-1])


def auc(data):
    xs, ys = data
    area = 0
    for i in range(len(xs)-1):
        area += (xs[i+1] - xs[i]) * (ys[i+1] + ys[i-1]) / 2  # trapezoid
    return area


def get_ref_acc(model):
    acc = {}
    for dataset in datasets:
        data_obj = Data(model, dataset, 'two_stage', testset=testset)
        acc[dataset] = data_obj.layer_acc[-1]
    return acc


def plot_acc_by_layer(axis, data, ref_acc):
    axis.set_xlim(1, data.size)

    if data.routine.endswith('lte'):
        # lte acc
        x, y = data.etp_acc
        if RELATIVE:
            y = list(map(lambda x: 100*x/ref_acc[dataset], y))
        axis.plot(x, y, 'o-', color=darkcolors[data.routine],
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
            y = data.layer_acc
            if RELATIVE:
                y = list(map(lambda x: 100*x/ref_acc[dataset], y))
            axis.plot(range(1, 1+len(data.layer_acc)), y,
                      'o-', color=darkcolors[data.routine],
                      label='\\textsc{' + data.formal_routine + '}',
                      linewidth=1, markersize=2)

        # entropy-based acc
        if (
                plot_target == 1 and
                (
                        (data.model == model and data.routine != 'limit') or
                        (data.model != model and data.routine == 'all_alternate')
                )
        ):
            color = darkcolors[data.routine]
            label = '\\textsc{' + data.formal_routine + '}'
            if data.model != model:
                color = color_pool[0]
                label = '\\textsc{db+alt}'
            x, y = data.etp_acc
            if RELATIVE:
                y = list(map(lambda x: 100*x/ref_acc[dataset], y))
            axis.plot(x, y, 'o-', color=color,
                      label=label,
                      linewidth=1, markersize=2)


# distilbert_data = DistilbertData()
for i_dataset, dataset in enumerate(datasets):
    if rna:
        if i_dataset<2:
            model = 'roberta-base'
        else:
            model = 'albert-base'

    if RELATIVE:
        ref_acc = get_ref_acc(model)
    else:
        ref_acc = {}
    dataset_axis = axes[i_dataset]

    # if plot_target==1 and model=='bert-base':
    #     dataset_axis.scatter(
    #         distilbert_data.saving * 12,
    #         distilbert_data.acc[dataset] / 100,
    #         s=100,  # size
    #         marker='+',
    #         # label='\\textsc{dist}'
    #     )

    for i_routine, routine in enumerate(routines):
        try:
            data_obj = Data(model, dataset, routine, testset=testset)
            plot_acc_by_layer(
                dataset_axis,
                data_obj,
                ref_acc
            )
            if model == 'bert-base' and plot_target == 1:
                db_data_obj = Data('distilbert-base', dataset, routine, testset=testset)
                plot_acc_by_layer(
                    dataset_axis,
                    db_data_obj,
                    ref_acc
                )
        except FileNotFoundError:
            pass

    try:
        dataset_axis.set_xlim(0, data_obj.size+1)

        if (
                i_dataset == 0 and
                (
                    (model in ['bert-base', 'roberta-base']) or
                    (model in ['bert-large', 'roberta-large'] and plot_target == 0)
                )
        ):
            dataset_axis.legend(
                bbox_to_anchor=(-0.2, 1.2, 2.5, 0.1),
                # xoffset, yoffset, width, height
                loc='upper left',
                ncol=100,
                mode="expand",
                borderaxespad=0.,
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
        dataset_axis.set_ylabel("Score" if not RELATIVE else "Relative Score (\\%)")
    except NameError:
        pass

if rna:
    model = 'rna-base'

if plot_target == 2:
    plt.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.9)
else:
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
plt.show()
# plt.savefig(f"{model}-{plot_target_name[plot_target]}.pdf")
