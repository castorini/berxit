import sys
import numpy as np
from matplotlib import pyplot as plt
from get_data import Data

model, rt1, rt2 = sys.argv[1:4]  # the two routines to compare auc

testset = False
if len(sys.argv)>3 and sys.argv[3] == 'testset':
    testset = True

if model == 'bert-base':
    datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]
else:
    datasets = ["RTE", "MRPC"]


class PieceWise:
    def __init__(self):
        self.pieces = 0
        self.k = []
        self.b = []
        self.xs = []

    def fit(self, _xs, _ys):
        # deduplicate first
        xs = []
        ys = []
        for i in range(len(_xs)):
            if len(xs) == 0 or _xs[i] != xs[-1]:
                xs.append(_xs[i])
                ys.append(_ys[i])

        self.pieces = len(xs) - 1
        self.xs = xs.copy()
        assert len(xs) == len(ys)
        assert len(xs) >= 2
        for i in range(self.pieces):
            self.k.append(
                (ys[i+1]-ys[i]) / (xs[i+1]-xs[i])
            )
            self.b.append(
                ys[i+1] - self.k[-1] * (xs[i+1] - xs[i])
            )

    def __call__(self, x):
        if type(x) is np.ndarray:
            return [self(xi) for xi in x]
        for i in range(self.pieces):
            if x >= self.xs[i] and x <= self.xs[i+1]:
                return self.k[i] * (x-self.xs[i]) + self.b[i]
        raise NotImplementedError()

    def get_piece_index(self, left, right):
        for i in range(self.pieces):
            if self.xs[i] <= left and right <= self.xs[i+1]:
                return i
        return -1


def get_cross_point(pw1, pw2, left, right):
    piece_index1 = pw1.get_piece_index(left, right)
    piece_index2 = pw2.get_piece_index(left, right)
    k1, b1, x1 = pw1.k[piece_index1], pw1.b[piece_index1], pw1.xs[piece_index1]
    k2, b2, x2 = pw2.k[piece_index2], pw2.b[piece_index2], pw2.xs[piece_index2]
    return (b2 - b1 - k2*x2 + k1*x1) / (k1 - k2)


def cal_diff(xs, y1, y2, pw1, pw2):
    adv1, adv2 = 0, 0
    prev_higher = None
    for i, x in enumerate(xs):
        if i == len(xs)-1:
            break
        cross = False  # whether the two lines cross here
        if y1[i+1] > y2[i+1]:
            if prev_higher == 2:
                cross = True
            else:
                adv1 += (y1[i] + y1[i+1]) * (xs[i+1] - x) / 2
                adv1 -= (y2[i] + y2[i+1]) * (xs[i+1] - x) / 2
            prev_higher = 1
        elif y2[i+1] > y1[i+1]:
            if prev_higher == 1:
                cross = True
            else:
                adv2 += (y2[i] + y2[i+1]) * (xs[i+1] - x) / 2
                adv2 -= (y1[i] + y1[i+1]) * (xs[i+1] - x) / 2
            prev_higher = 2
        elif y1[i+1] == y2[i+1]:
            print('equal!')
            raise NotImplementedError()

        if cross:
            cross_point = get_cross_point(pw1, pw2, x, xs[i+1])
            if y1[i] > y2[i]:
                adv1 += (y1[i] - y2[i]) * (cross_point - x) / 2
                adv2 += (y2[i+1] - y1[i+1]) * (xs[i+1] - cross_point) / 2
            else:
                adv2 += (y2[i] - y1[i]) * (cross_point - x) / 2
                adv1 += (y1[i+1] - y2[i+1]) * (xs[i+1] - cross_point) / 2

    return adv1, adv2


fig, axes = plt.subplots(2, 1)
for ds in datasets:
    print(ds)
    ref_data = Data(model, ds, 'two_stage', testset=testset)
    base_acc = ref_data.layer_acc[-1]

    x1 = [x[1]['mean_exit'] for x in Data(model, ds, rt1, testset=testset).etp_data]
    x2 = [x[1]['mean_exit'] for x in Data(model, ds, rt2, testset=testset).etp_data]
    y1 = [x[1]['acc'] for x in Data(model, ds, rt1, testset=testset).etp_data]
    y2 = [x[1]['acc'] for x in Data(model, ds, rt2, testset=testset).etp_data]
    axes[0].plot(x1, y1, label='JOINT')
    axes[0].plot(x2, y2, label='ALT')

    pw1 = PieceWise()
    pw1.fit(x1, y1)
    pw2 = PieceWise()
    pw2.fit(x2, y2)

    merge_x = sorted(list(set(x1 + x2)))
    y1_after = [pw1(x) for x in merge_x]
    y2_after = [pw2(x) for x in merge_x]

    plot_x = np.linspace(1, 12, 200)
    axes[1].plot(merge_x, y1_after, 'o-')
    axes[1].plot(merge_x, y2_after, 'o-')

    print(cal_diff(merge_x, y1_after, y2_after, pw1, pw2))

