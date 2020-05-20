import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('font', size=20)
matplotlib.rc('text', usetex=True)

target = sys.argv[1]  # entropy or uncertainty

if target == 'entropy':
    dataset = 'MRPC'
    np_data = np.load(f'saved_models/bert-base/{dataset}/all_alternate-42/entropy_distri.npy')
    box_data = list(np_data.transpose())
    title = 'BERT\\textsubscript{\\textsc{base}} : MRPC'
    acc_data = np.load(f'saved_models/bert-base/{dataset}/all_alternate-42/each_layer.npy')
else:
    dataset = 'STS-B'
    box_data = [[] for _ in range(12)]
    with open(f'saved_models/bert-base/{dataset}/all_alternate-lte-42/uncertainty.txt') as fin:
        for line in fin:
            a = line.split('\t')
            for i in range(12):
                box_data[i].append(float(a[i]))
    title = 'BERT\\textsubscript{\\textsc{base}} : STS-B'
    acc_data = np.load(f'saved_models/bert-base/{dataset}/all_alternate-42/each_layer.npy')

fig, axes = plt.subplots(1, 1, figsize=[8, 4])


sns.boxplot(
    data=box_data,
    color='navajowhite'
)
axes.set_xticklabels(range(1, 13))
axes.set_xlabel('Exit Layer')
axes.set_ylabel(target.capitalize())
axes.set_title(title)

twin_axes = axes.twinx()
twin_axes.plot(acc_data, 'o-')
twin_axes.set_ylabel("Score")

plt.tight_layout()
# plt.show()
plt.savefig('box-'+target+'.pdf')