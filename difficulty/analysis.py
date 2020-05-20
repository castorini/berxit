import os
import sys
import numpy as np

model = 'bert-base'
routine = 'all_alternate'
dataset = sys.argv[1]
chosen_layer = 5

filters = {
    'QQP': lambda x: x.split('\t')[3:],
    'MRPC': lambda x: x.split('\t')[3:5] + [x[0]]
}  # returns [Q1, Q2, score]

def print_table(model, dataset, chosen_layer, filter):
    etp = np.load(f'plotting/saved_models/{model}/{dataset}/'
                  f'{routine}-42/entropy_distri.npy')
    pred = np.load(f'plotting/saved_models/{model}/{dataset}/'
                   f'{routine}-42/prediction_layer{chosen_layer}.npy')
    with open(f'difficulty/{dataset}.tsv', 'w') as fout:
        with open(f'../GLUE/{dataset}/dev.tsv') as fin:
            fin.readline()
            for i in range(len(etp)):
                line = fin.readline().strip()
                instance = filter(line)
                if len(instance)<3:
                    # only happens once
                    print(i, line)
                    line = fin.readline().strip()
                    instance = filter(line)

                print('{:.3f}'.format(etp[i, chosen_layer]), end='\t', file=fout)
                print('label:' + instance[2], end='\t', file=fout)
                print(f'pred:{pred[i]}', file=fout)
                print(instance[0], file=fout)
                print(instance[1], file=fout)
                print(file=fout)

print_table(model, dataset, chosen_layer, filters[dataset])
