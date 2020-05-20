import sys
from get_data import Data, DistilbertData, RawBertData

model, routine = sys.argv[1:3]

testset = False
if len(sys.argv)>3 and sys.argv[3] == 'testset':
    testset = True
datasets = ["RTE", "MRPC", "SST-2", "QNLI", "QQP", "MNLI"]

dbdata = DistilbertData()
rbdata = RawBertData(size=model.split('-')[1])

for ds in datasets:
    data = Data(model, ds, routine, testset=testset)
    col = data.etp_data
    # show data
    print(ds)
    print('etp\tlayer\tRlayer\tacc\tRacc\tdrop')
    base_acc = rbdata.acc[ds]
    base_layer = rbdata.layers
    if model.startswith('distil'):
        base_layer = 6
        shrink = dbdata.saving
        print('{}\t{:.1f}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
            'dev',
            6,
            shrink,
            dbdata.acc[ds],
            dbdata.acc[ds] / base_acc,
            dbdata.acc[ds] - base_acc,
        ))
    else:
        shrink = 1

    for line in col[::-1]:
        acc = line[1]['acc'] * 100
        print('{}\t{:.1f}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
            line[0],
            line[1]['mean_exit'],
            line[1]['mean_exit'] / base_layer * shrink,
            acc,
            acc / base_acc,
            acc -base_acc,
            ))
    print()