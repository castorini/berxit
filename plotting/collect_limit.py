import sys
import shutil
import numpy as np


path, model_size = sys.argv[1:]

layers = 12 if model_size == 'base' else 24

col = []
for i in range(layers):
    try:
        data = np.load(path+f'/layer-{i}.npy')
    except FileNotFoundError:
        print('File Not Found at Layer ' + str(i))
        exit(1)
    col.append(float(data))

try:
    shutil.copyfile(path+'/each_layer.npy', path+'/each_layer.npy.backup')
    np.save(path+'/each_layer.npy', np.array(col))
except FileNotFoundError:
    pass
