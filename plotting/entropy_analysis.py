import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

model = sys.argv[1]
num_layers = 12 if model.endswith("base") else 24

analyze_entropy_per_layer = False
if analyze_entropy_per_layer:

    fig, axes = plt.subplots(2, 2, figsize=[20, 20])
    axes = axes.reshape(-1)
    for i, dataset in enumerate(["QNLI", "SST-2", "MRPC", "RTE"]):
        data = np.load(
            "saved_models/{}/{}/two_stage-42/entropy_distri.npy".format(model, dataset),
            allow_pickle=True)
        axes[i].set_title(dataset)

        # violin
        # axes[i].violinplot(data)

        # tendancy curve
        selection = np.random.choice(len(data), 10, replace=False)
        for sel in selection:
            axes[i].plot(data[sel])

    plt.show()
    exit(0)
    # plt.tight_layout()
    # plt.savefig("entropy_violin_{}.pdf".format(model))

dataset, routine = sys.argv[2:]
filepath = f"saved_models/{model}/{dataset}/{routine}-42/"
entropy_col = np.load(filepath+"entropy_distri.npy")
maxlogit_col = np.load(filepath+"maxlogit_distri.npy")
correct_col = []
for i in range(num_layers - 1):
    correct_col.append(np.load(filepath+"correctness_layer{}.npy".format(i)))
correct_col.append(np.load(filepath+"correctness_layer-1.npy"))
correct_col = np.transpose(correct_col)
n = len(entropy_col)

analyze_entropy_and_correctness = True
if analyze_entropy_and_correctness:

    fig, axes = plt.subplots(12, 12, figsize=[20, 20])
    axes = axes.reshape(-1)
    def plot_unit(axis, ent, cor, mlog):
        axis.plot(ent)
        axis.plot(cor*1.1)
        axis.plot(mlog)
        # axis.set_ylim(0, 0.7)
        axis.set_yticks([])

    i = 0
    easy = 0
    wrong = 0
    for x in range(n):
        if entropy_col[x][0]<0.05:
            easy += 1
            if correct_col[x][0]==0:
                wrong += 1
        if i<144:
            plot_unit(axes[i], entropy_col[x], correct_col[x], maxlogit_col[x])
            i += 1

    print(n, easy, wrong)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"entropy_trend-{model}-{dataset}-{routine}")


analyze_correlation = False
if analyze_correlation:
    fig, axes = plt.subplots(1, 1)
    points = [[], []]
    for i, x in enumerate(entropy_col):
        for j in range(12):
            points[int(correct_col[i][j])].append(x[j])

    plt.hist(points, bins=20, label=["Incorrect", "Correct"], rwidth=1)
#     axes.set_xlim(0,0.8)
    #axes.set_yticks([])
    plt.xlabel("Entropy")
    plt.legend()
    plt.title("{}@{}".format(dataset, routine))
    # plt.show()
    plt.tight_layout()
    plt.savefig("{}_density_{}.pdf".format(dataset, routine))
