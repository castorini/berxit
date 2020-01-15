
datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
sizes = ["2.5k", "3.5k", "67k", "108k", "363k", "392k"]
routines = ["two_stage", "all", "self_distil"]#, "neigh_distil", "layer_wise", "divide"]
M, N = 2, len(datasets)//2
fig, axes = plt.subplots(M, N, figsize=[N*4, M*4])
axes.reshape([-1])

def plot_acc_by_layer(axis, datafile, title):
    for j, d in enumerate(data):
        axis.plot(d, 'o-', label=routines[j])
    axis.legend(loc='lower right')
    axis.set_title(title)
    axis.set_xlabel("Exit layer")
    axis.set_ylabel("Score")




for i in range(len(datasets)):
    data = []
    for j in range(len(routines)):
        if i!=2 and j>2:
            continue
        try:
            data.append(
                np_load("saved_models/bert-base/{}/{}-42/each_layer.npy".format(datasets[i], routines[j]))
            )
        except:
            pass
    plot_acc_by_layer(
        axes[i//N][i%N],
        data,
        title=datasets[i]+' ('+sizes[i]+')'
    )
plt.tight_layout()
# plt.show()
plt.savefig("Bert-base.pdf")