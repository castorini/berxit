import sys
import shutil
import numpy as np


path, model_size = sys.argv[1:]

layers = 12 if model_size == 'base' else 24

col = []
for i in range(layers-1):
    try:
        data = np.load(path+f'/layer-{i}.npy')
    except FileNotFoundError:
        print('File Not Found at Layer ' + str(i))
        exit(1)
    col.append(float(data))
col.append(float(np.load(path+'/../two_stage-42/each_layer.npy')[-1]))

try:
    shutil.copyfile(path+'/each_layer.npy', path+'/each_layer.npy.backup')
    np.save(path+'/each_layer.npy', np.array(col))
except FileNotFoundError:
    pass
