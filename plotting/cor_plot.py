import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RELATIVE = True

matplotlib.rc('font', size=20)
matplotlib.rc('text', usetex=True)

fig, axes = plt.subplots(2, 1, figsize=[8, 8])

def plot_dataset(axes, dataset):
    title = 'BERT\\textsubscript{\\textsc{base}} : ' + dataset

    acc_data = np.load(f'saved_models/bert-base/{dataset}/all_alternate-42/each_layer.npy')
    cor_data = np.load(f'../difficulty/{dataset}-cor.npy')

    axes.plot(range(1, 13), cor_data[:, 1], 'o-', color='r', label='pos.')
    axes.plot(range(1, 13), cor_data[:, 2], 'o-', color='b', label='neg.')
    axes.hlines(y=0, xmin=1, xmax=12, linestyle=':', color='k')

    axes.set_xticks(range(1, 13))
    axes.set_xticklabels(range(1, 13))
    axes.set_xlabel('Exit Layer')
    axes.set_ylabel('Pearson Correlation')
    axes.set_yticks([-0.5, 0, 0.5])
    axes.set_yticklabels([-0.5, 0, 0.5])
    lim = 0.8 if dataset=='MRPC' else 0.5
    axes.set_ylim(-lim, lim)
    axes.set_title(title)
    axes.legend(loc='upper left' if dataset=='MRPC' else 'lower right')

    twin_axes = axes.twinx()
    if RELATIVE:
        base = np.load(f'saved_models/bert-base/{dataset}/two_stage-42/each_layer.npy')[-1]
        acc_data = list(map(lambda x: 100*x/base, acc_data))
    twin_axes.plot(range(1, 13), acc_data, color='k')
    twin_axes.set_ylabel("Relative Score (\\%)")

plot_dataset(axes[0], 'MRPC')
plot_dataset(axes[1], 'QQP')

plt.tight_layout()
# plt.show()
plt.savefig('cor.pdf')
