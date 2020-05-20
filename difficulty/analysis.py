import os
import sys
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

model = 'bert-base'
routine = 'all_alternate'
dataset = sys.argv[1]
# chosen_layer = 3

filters = {
    'QQP': lambda x: x.split('\t')[3:],
    'MRPC': lambda x: x.split('\t')[3:5] + [x[0]],
    'QNLI': lambda x: x.split('\t')[1:]
}  # returns [Q1, Q2, score]


def obtain_data(model, dataset, chosen_layer, filter):
    etp = np.load(f'plotting/saved_models/{model}/{dataset}/'
                  f'{routine}-42/entropy_distri.npy')
    pred = np.load(f'plotting/saved_models/{model}/{dataset}/'
                   f'{routine}-42/prediction_layer{chosen_layer}.npy')

    if dataset in ['QQP', 'MRPC']:
        label_processor = int
    else:
        label_processor = lambda x: 1 if x=='entailment' else 0

    col = []
    with open(f'../GLUE/{dataset}/dev.tsv') as fin:
        fin.readline()
        for i in range(len(etp)):
            line = fin.readline().strip()
            instance = filter(line)
            if len(instance) < 3:
                # only happens once in QQP
                # print(i, line)
                line = fin.readline().strip()
                instance = filter(line)

            col.append([
                instance[0],
                instance[1],  # the two sentences
                label_processor(instance[2]),  # label
                pred[i],  # prediction
                etp[i, chosen_layer],  # entropy
            ])

    return col


def print_table(dataset, col):
    with open(f'difficulty/{dataset}.tsv', 'w') as fout:
        for c in col:
            print('{:.3f}'.format(c[4]), end='\t', file=fout)
            print(f'label:{c[2]}', end='\t', file=fout)
            print(f'pred:{c[3]}', file=fout)
            print(c[0], file=fout)
            print(c[1], file=fout)
            print(file=fout)


sample_rate = 5  # one point every this many
def bleu_correlation(col, chosen_layer):
    etp, bleu = [[] for _ in range(3)], [[] for _ in range(3)]
    for i, c in enumerate(col):
        if i % sample_rate == 0:
            curr_etp = c[4]
            curr_bleu = sentence_bleu([c[0]], c[1])
            etp[0].append(curr_etp)
            bleu[0].append(curr_bleu)
            if c[3]==1:
                etp[1].append(curr_etp)
                bleu[1].append(curr_bleu)
            else:
                etp[2].append(curr_etp)
                bleu[2].append(curr_bleu)
    print(f'layer={chosen_layer}')
    for j in range(3):
        if len(etp[j]) < 2:
            print('N/A', end='\t')
        else:
            print('{:.3f}'.format(pearsonr(etp[j], bleu[j])[0]), end='\t')
    print()
    plt.cla()
    plt.scatter(etp[1], bleu[1], s=1, c='r')  # positive
    plt.scatter(etp[2], bleu[2], s=1, c='b')  # negative
    plt.xlabel('Entropy')
    plt.ylabel('BLEU')
    plt.title(f'layer {chosen_layer}')

    plt.savefig(f'difficulty/layer{chosen_layer}.png')

for chosen_layer in range(11):
    col = obtain_data(model, dataset, chosen_layer, filters[dataset])
    bleu_correlation(col, chosen_layer)
