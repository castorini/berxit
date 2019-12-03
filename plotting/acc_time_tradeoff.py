import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def np_load(fname):
    return np.load(fname, allow_pickle=True)


get_entropy = lambda x: float(x[x.index('entropy_')+8:x.index('.npy')])


def get_files(model, dataset):
    filepath = f"saved_models/{model}/{dataset}/"
    nextlevel = os.listdir(filepath)
    if len(nextlevel)>1:
        raise ValueError(filepath + " has more than 1 child folder!")
    filepath += nextlevel[0] + '/'
    entropy_files = [x for x in os.listdir(filepath) if "entropy" in x]
    entropy_files.sort(key=get_entropy)
    return [
        filepath + "each_layer.npy",
        [filepath + x for x in entropy_files]
    ]


def get_saving(lst):
    new_lst = [100*(lst[0]-x)/lst[0] for x in lst]
    return new_lst


def plot_acc_by_layer(axis, data, labels, flags):
    styles = ['o-', '+-']
    for i, d in enumerate(data):
        xticks = np.arange(1, len(d)+1)
        axis.plot(xticks, d, styles[i], label=labels[i])
        axis.xaxis.set_ticks(xticks)
    axis.legend()
    axis.set_xlabel("Exit Layer")
    axis.set_ylabel("F1 score" if flags["f1"] else "Accuracy")


def plot_acc_time_tradeoff(axis, data, labels, flags):
    time, acc = data
    styles = ['x-', '+-']
    for i in range(2):
        saving = get_saving(time[i])
        plot_data = list(zip(saving, acc[i]))
        plot_data.sort()
        axis.plot(*list(zip(*plot_data)), styles[i], label=labels[i])
    axis.legend()
    axis.set_xlabel("Runtime saving (%)")
    axis.set_ylabel("F1 score" if flags["f1"] else "Accuracy")


def plot_exit_samples_by_layer(axis, data, title):
    entropy, samples_layer = data
    n = len(entropy)
    for i, (en, sa) in enumerate(zip(entropy, samples_layer)):
        if not (i<=1 or i==n-1 or i==n/2 or i==(n-1)/2):
            continue
        plot_x = np.arange(1, len(sa)+1)
        plot_y = [sa[x]+1 for x in plot_x]
        axis.plot(plot_x, plot_y, '+-', label=str(en))
    axis.xaxis.set_ticks(plot_x)
    axis.set_title(title)
    axis.set_yscale('log')
    axis.set_xlabel('Exit layer')
    axis.set_ylabel('Number of samples')
    axis.legend()



dataset = sys.argv[1]
each_layer_acc_data = []
entropy_data = []
time_data = []
acc_data = []
samples_layer_data = []

large_flag = ""
if len(sys.argv)>2 and sys.argv[2]=="large":
    models = ['bert-large', 'roberta-large']
    large_flag = "-large"
else:
    models = ['bert-base', 'roberta-base']

for model in models:
    each_layer_file, entropy_files = get_files(model, dataset)

    each_layer_acc_data.append(np_load(each_layer_file))

    model_entropy_data = []
    model_time_data = []
    model_acc_data = []
    model_samples_layer_data = []

    for ef in entropy_files:
        ef_data = np_load(ef)
        model_time_data.append(ef_data[1])
        model_acc_data.append(ef_data[3])
        model_samples_layer_data.append(ef_data[0])
        model_entropy_data.append(get_entropy(ef))

    entropy_data.append(model_entropy_data)
    time_data.append(model_time_data)
    acc_data.append(model_acc_data)
    samples_layer_data.append(model_samples_layer_data)

fig, axes = plt.subplots(1, 1)

# plot_acc_by_layer(axes,
#                   each_layer_acc_data,
#                   labels=models,
#                   flags={"f1": dataset=="MRPC"})
# plt.grid(which='both', axis='both')
# plt.tight_layout()
# plt.savefig(f"{dataset}{large_flag}-acc_level.pdf")
# plt.cla()
#
#
# plot_acc_time_tradeoff(axes,
#                        [time_data, acc_data],
#                        labels=models,
#                        flags={"f1": dataset=="MRPC"})
#
# plt.grid(which='both', axis='both')
# plt.tight_layout()
# plt.savefig(f"{dataset}{large_flag}-tradeoff.pdf")
# plt.cla()

fig, axes = plt.subplots(2, 1)
for i in range(2):
    plot_exit_samples_by_layer(axes[i],
                               [entropy_data[i], samples_layer_data[i]],
                               title=models[i]+' @ '+dataset)
    plt.grid(which='both', axis='both')
plt.tight_layout()
plt.savefig(f"{dataset}{large_flag}-sample_layer.pdf")
plt.cla()


