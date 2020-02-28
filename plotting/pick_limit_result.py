import os
import numpy as np

for log_id in range(751, 767):
    if log_id in [753, 760, 765]:
        continue
    print(log_id)
    with open("../logs/{}.log".format(log_id)) as fin:
        lines = fin.readlines()
    for line in lines:
        if "Namespace" in line:
            flag1 = line.find("output_dir")
            flag2 = line.find("-42',")
            location = line[flag1+12 : flag2+3]
    num_layer = 12 if 'base' in location else 24
    metric = ' ' + ('f1' if ('MRPC' in location or 'QQP' in location) else 'acc') + ' ='
    # print(num_layer, metric, location)
    results = []
    final_result = None
    for line in lines:
        if metric in line:
            number = float(line[line.find(metric)+len(metric):])
            if final_result is None:
                final_result = number
            else:
                results.append(number)
    results.append(final_result)
    if not os.path.exists(location):
        os.makedirs(location)
    np.save(location+"/each_layer.npy", np.array(results))
