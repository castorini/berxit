import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns


def np_load(fname):
    return np.load(fname, allow_pickle=True)


get_entropy = lambda x: float(x[x.index('entropy_')+8:x.index('.npy')])
def get_entropy(x):
    try:
        return float(x[x.index('entropy_')+8:x.index('.npy')])
    except ValueError:
        return False


def get_files(model, dataset):
    filepath = f"saved_models/{model}/{dataset}/"
    nextlevel = os.listdir(filepath)
    nextlevel = [x for x in nextlevel if ".zip" not in x]
    idx = nextlevel.index('two_stage-42')
    filepath += nextlevel[idx] + '/'
    entropy_files = [x for x in os.listdir(filepath) if "entropy" in x and get_entropy(x)]
    entropy_files.sort(key=get_entropy)
    return [
        filepath + "each_layer.npy",
        [filepath + x for x in entropy_files]
    ]


def get_saving(lst):
    new_lst = [100*(lst[0]-x)/lst[0] for x in lst]
    return new_lst


def plot_acc_time_tradeoff(axis, data, labels, flags):
    time, acc = data
    styles = ['x-', 'x-']
    for i in range(2):
        saving = get_saving(time[i])
        plot_data = list(zip(saving, acc[i]))
        plot_data.sort()
        axis.plot(*list(zip(*plot_data)), styles[i], label=labels[i],
                  linewidth=2.5, markersize=15)
    axis.legend()
    axis.set_title(flags["title"], fontweight='bold')
    axis.set_xlabel("Runtime Speedup (%)")
    axis.set_ylabel(("F1 Score" if flags["f1"] else "Accuracy")+" (%)")


def plot_acc_by_layer(axis, data, labels, flags):
    styles = ['o-', 'x-']
    for i, d in enumerate(data):
        xticks = np.arange(1, len(d)+1)
        d = [100*x for x in d]
        axis.plot(xticks, d, styles[i], label=labels[i],
                  linewidth=2.5, markersize=10)
        # axis.xaxis.set_ticks(xticks)
    axis.legend()
    axis.set_title(flags["title"], fontweight='bold')
    axis.set_xlabel("Exit Layer")
    axis.set_ylabel(("F1 Score" if flags["f1"] else "Accuracy")+" (%)")


def plot_ers_and_time(axis, data, labels, flags):
    _, err, time = data
    for i in range(2):
        time_lst = get_saving(time[i])
        ers = [100*(1-x) for x in err[i]]
        axis.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
        axis.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
        axis.plot(ers, time_lst, '+-', label=labels[i],
                      linewidth=2.5, markersize=10)
        axis.legend()
        axis.set_xlabel("Expected Speedup (%)")
        axis.set_ylabel("Measured Speedup (%)")
        axis.set_title(flags["title"], fontweight='bold')


def plot_exit_samples_by_layer(axis, data, title):
    entropy, samples_layer = data
    n = len(entropy)
    for i, (en, sa) in enumerate(zip(entropy, samples_layer)):
        if not (i<=1 or i==n-1 or i==n/2 or i==(n-1)/2):
            continue
        plot_x = np.arange(1, len(sa)+1)
        plot_y = [sa[x]+1 for x in plot_x]
        all_samples = sum(plot_y)
        plot_y = [100*(y/all_samples) for y in plot_y]
        axis.plot(plot_x, plot_y, '+-', label=str(en),
                  linewidth=2.5, markersize=10)
    # axis.xaxis.set_ticks(plot_x)
    axis.set_title(title, fontweight='bold')
    axis.set_yscale('log')
    axis.set_xlabel('Exit Layer')
    axis.set_ylabel('Fraction of Dataset')
    axis.yaxis.set_major_locator(plticker.FixedLocator([1, 10, 100]))
    axis.yaxis.set_ticklabels(['1%', '10%', '100%'])
    axis.legend()


def show_results(data):
    entropy, time, acc, err = data
    for i in range(2):
        saving = get_saving(time[i])
        for j in range(len(entropy[i])):
            print('{}\t{:.2f}\t{:.2f}\t{:.0f}\t{:.2f}\t{:.1f}'.format(
                entropy[i][j],
                acc[i][j],
                acc[i][j] - acc[i][0],
                saving[j],
                time[i][j],
                100*(1-err[i][j])
            ))
        print()



dataset = sys.argv[1]
each_layer_acc_data = []
entropy_data = []
time_data = []
acc_data = []
err_data = []
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
    model_err_data = []
    model_samples_layer_data = []

    for ef in entropy_files:
        ef_data = np_load(ef)
        model_time_data.append(ef_data[1])
        model_acc_data.append(ef_data[3]*100)
        model_err_data.append(ef_data[2])
        model_samples_layer_data.append(ef_data[0])
        model_entropy_data.append(get_entropy(ef))

    entropy_data.append(model_entropy_data)
    time_data.append(model_time_data)
    acc_data.append(model_acc_data)
    err_data.append(model_err_data)
    samples_layer_data.append(model_samples_layer_data)


show_results([entropy_data, time_data, acc_data, err_data])

exit(0)


sns.set(style='whitegrid', font_scale=2.5)
fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
plot_acc_time_tradeoff(axes,
                       [time_data, acc_data],
                       labels=models,
                       flags={"f1": dataset=="MRPC",
                              "title": dataset}
                       )

plt.tight_layout(pad=0)
plt.savefig(f"{dataset}{large_flag}-tradeoff.pdf")
plt.cla()





sns.set(style='whitegrid', font_scale=2.5)
fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])
plot_acc_by_layer(axes,
                  each_layer_acc_data,
                  labels=models,
                  flags={"f1": dataset == "MRPC",
                         "title": dataset}
                  )
plt.tight_layout(pad=0)
plt.savefig(f"{dataset}{large_flag}-acc_level.pdf")
plt.cla()



sns.set(style='white', font_scale=2.5)
fig, axes = plt.subplots(1, 1, figsize=[6.4, 6.4])  #6.4 for a quarter textwidth
plot_ers_and_time(axes,
                  [entropy_data, err_data, time_data],
                  labels=models,
                  flags={"title": dataset}
                  )
plt.grid(axis='both', which='major')
plt.tight_layout(pad=0)
plt.savefig(f"{dataset}{large_flag}-ers_real.pdf")
plt.cla()








sns.set(style='white', font_scale=1.25)
fig, axes = plt.subplots(2, 1, figsize=[6.4, 6.4*0.8])
for i in range(2):
    plot_exit_samples_by_layer(axes[i],
                               [entropy_data[i], samples_layer_data[i]],
                               title=models[i]+' @ '+dataset)
    axes[i].grid(which='both', axis='y')
plt.tight_layout(pad=0)
plt.savefig(f"{dataset}{large_flag}-sample_layer.pdf")
plt.cla()

