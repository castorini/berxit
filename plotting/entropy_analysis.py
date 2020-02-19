import sys
import numpy as np
from matplotlib import pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=[20, 20])
axes = axes.reshape(-1)
model = sys.argv[1]

for i, dataset in enumerate(["QNLI", "SST-2", "MRPC", "RTE"]):
    data = np.load(
        "saved_models/{}/{}/two_stage-42/entropy_distri.npy".format(model, dataset))
    axes[i].set_title(dataset)

    # violin
    # axes[i].violinplot(data)

    # tendancy curve
    selection = np.random.choice(len(data), 10, replace=False)
    for sel in selection:
        axes[i].plot(data[sel])

plt.show()
# plt.tight_layout()
# plt.savefig("entropy_violin_{}.pdf".format(model))
