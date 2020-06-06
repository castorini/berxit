import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

RELATIVE = True
model = 'bert-base'
layers = 12 if model.endswith('base') else 24

matplotlib.rc('font', size=20)
matplotlib.rc('text', usetex=True)

target = sys.argv[1]  # entropy or uncertainty

if target == 'entropy':
    dataset = 'MRPC'
    np_data = np.load(f'saved_models/{model}/{dataset}/all_alternate-42/entropy_distri.npy')
    box_data = list(np_data.transpose())
    title = 'BERT\\textsubscript{\\textsc{base}} : MRPC'
    acc_data = np.load(f'saved_models/{model}/{dataset}/all_alternate-42/each_layer.npy')
else:
    dataset = 'STS-B'
    box_data = [[] for _ in range(layers)]
    with open(f'saved_models/{model}/{dataset}/all_alternate-lte-42/uncertainty.txt') as fin:
        for line in fin:
            a = line.split('\t')
            for i in range(layers):
                box_data[i].append(float(a[i]))
    title = 'BERT\\textsubscript{\\textsc{base}} : STS-B'
    acc_data = np.load(f'saved_models/{model}/{dataset}/all_alternate-42/each_layer.npy')

fig, axes = plt.subplots(1, 1, figsize=[8, 3])

sns.boxplot(
    data=box_data,
    color='navajowhite'
)
axes.set_xticklabels(range(1, 1+layers))
axes.set_xlabel('Exit Layer')
axes.set_ylabel(target.capitalize())
axes.set_title(title)

twin_axes = axes.twinx()
if RELATIVE:
    base = np.load(f'saved_models/{model}/{dataset}/two_stage-42/each_layer.npy')[-1]
    acc_data = list(map(lambda x: 100*x/base, acc_data))
twin_axes.plot(acc_data, 'o-')
twin_axes.set_ylabel("Relative Score (\\%)")

plt.tight_layout(pad=0.5)
plt.show()
# plt.savefig('box-'+target+'.pdf')
