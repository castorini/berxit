import sys
import os
import time
import subprocess
import numpy as np
import comet_ml as cm



log_folder = "logs"
existed_experiments = []
for x in os.listdir(log_folder):
    num, suffix = x.split('.')
    if num not in existed_experiments and num.isnumeric():
        existed_experiments.append(num)
filecount = len(existed_experiments)


args = sys.argv[1:]
if len(args)==7:
    flavor, model, dataset, seed, routine, entropy, inter = args
    HP = None
elif len(args)==8:
    flavor, model, dataset, seed, routine, entropy, inter, HP = args
else:
    raise ValueError("Wrong number of parameters")

# flavor: raw, train_highway, eval_highway
# model: bert-base, bert-large, roberta-base, roberta-large
# dataset: MRPC, SST-2, RTE, QNLI
# seed: 42, 9102, 4396
# routine: raw(?), two_stage(ACL), all(new)
# entropy: anything, -1 will evaluate all layers

entropy_selection = {
    # there might be additional ones for specific routines
    "bert-base": {
        "MRPC":  "0,0.1,0.3,0.5",  # testset
        # "MRPC":  "0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.55,0.57,0.6,0.65,0.68", # 0.6,0.61 for two_stage; 0.58 for alternating
        "SST-2": "0,0.005, 0.05, 0.1, 0.2",  # testset
        # "SST-2": "0,0.001,0.005,0.01,0.05,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.68,0.7", # 0.5,0.55,0.6,0.65 for two_stage
        "QNLI":  "0,0.06,0.15,0.35",  # testset
        # "QNLI":  "0,0.06,0.1,0.14,0.15,0.2,0.3,0.35", # 0.5,0.55,0.6,0.65 for two_stage
        "RTE":   "0,0.35, 0.4, 0.6", #  # testset
        # "RTE":   "0,0.2,0.25,0.3,0.35,0.4,0.6,0.65", # 0.68 for two_stage
        "QQP":   "0,0.01,0.05,0.10.3",  # testset
        # "QQP":   "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6", # 0.65 for two_stage
        "MNLI":  "0,0.05,0.2,0.5"  # testset
        # "MNLI":  "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.65"
    },
    "roberta-base": {
        # "MRPC":  "0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.3,0.5,0.55,0.6,0.7",
        "MRPC":  "0,0.01,0.04,0.5",  # testset
        # "SST-2": "0,0.001,0.01,0.1,0.2,0.3,0.5,0.55,0.6,0.7",
        "SST-2": "0,0.1,0.3,0.55",  # testset
        # "QNLI":  "0,0.06,0.1,0.2,0.25,0.45,0.5,0.55,0.7",
        "QNLI":  "0,0.1,0.25,0.5",  # testset
        # "RTE":   "0,0.03,0.05,0.25,0.3,0.45,0.5,0.7",
        "RTE":   "0,0.25,0.3,0.5",  # testset
        # "QQP":   "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        "QQP":   "0,0.1,0.3,0.4",  # testset
        # "MNLI":   "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7" # might need to run again for dev
        "MNLI":   "-0.5,0.05,0.4,0.8,1.05"  # testset
    },
    "bert-large": {
        # "MRPC":  "0",
        "MRPC":  "0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.68",
        "SST-2": "0,0.001,0.005,0.01,0.05,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.68",
        "QNLI":  "0.14,0.15,0.2,0.3,0.35",
        # "QNLI":  "0,0.06,0.1,0.14,0.15,0.2,0.3,0.35",
        "MNLI":   "0,0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5"
    },
    "roberta-large": {
        "MRPC":  "0,0.05,0.1,0.15,0.2,0.3,0.5,0.55,0.6,0.65,0.7",
        "SST-2": "0,0.001,0.01,0.1,0.2,0.3,0.5,0.55,0.6,0.65,0.7",
        "QNLI":  "0,0.06,0.1,0.2,0.25,0.45,0.5,0.55,0.6,0.7",
        "RTE":   "0,0.03,0.05,0.25,0.3,0.45,0.5,0.6,0.7",
        "QQP":   "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        "MNLI":  "0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    }
}


best_learning_rate = {
    "bert-base":{
            "CoLA" : "4e-5",
            "SST-2" : "2e-5",
            "MRPC" : "2e-5",
            "STS-B" : "5e-5",
            "QQP" : "2e-5",
            "MNLI" : "3e-5",
            "QNLI" : "2e-5",
            "RTE" : "3e-5",
            "WNLI" : "3e-5"
        },
    "bert-large":{
            "CoLA" : "2e-5",
            "SST-2" : "1e-5",
            "MRPC" : "2e-5",
            "STS-B" : "4e-5",
            "QQP" : "1e-5",
            "MNLI" : "1e-5",
            "QNLI" : "2e-5",
            "RTE" : "3e-5",
            "WNLI" : "2e-5"
        },
    "roberta-base":{
            "CoLA" : "4e-5",
            "SST-2" : "1e-5",
            "MRPC" : "2e-5",
            "STS-B" : "2e-5",
            "QQP" : "1e-5",
            "MNLI" : "1e-5",
            "QNLI" : "1e-5",
            "RTE" : "3e-5",
            "WNLI" : "5e-5"
        },
    "roberta-large":{
            "CoLA" : "1e-5",
            "SST-2" : "1e-5",
            "MRPC" : "1e-5",
            "STS-B" : "2e-5",
            "QQP" : "1e-5",
            "MNLI" : "1e-5",
            "QNLI" : "1e-5",
            "RTE" : "1e-5",
            "WNLI" : "3e-5"
        },
    "albert-base":{
            "CoLA" : "2e-5",
            "SST-2" : "2e-5",
            "MRPC" : "2e-5",
            "STS-B" : "2e-5",
            "QQP" : "2e-5",
            "MNLI" : "2e-5",
            "QNLI" : "2e-5",
            "RTE" : "2e-5",
            "WNLI" : "2e-5"
        },
}
best_epochs = {
    "bert-base": 3,
    "bert-large": 3,
    "roberta-base": 10,
    "roberta-large": 10,
    "albert-base": 1,  # just for debug!
}





def inter_run(script):
    subprocess.run(script, shell=True)


def submit_run(script, filecount):
    with open("slurm_submit.sh", 'w') as f:
        print(script, file=f)
    if os.environ["HOSTNAME"]=='v':
        subprocess.run("python v2_submit.py slurm_submit.sh " + str(filecount), shell=True)
    else:
        subprocess.run("python submit.py slurm_submit.sh " + str(filecount), shell=True)


def get_model_suffix(model):
    if model.startswith('bert'):
        return '-uncased'
    elif model.startswith('albert'):
        return '-v2'
    elif model.startswith('roberta'):
        return ''
    else:
        raise NotImplementedError()


script_template = {
    "raw":
        r"""python -um examples.run_glue \
    --model_type {} \
    --model_name_or_path {} \
    --task_name {} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /h/xinji/projects/GLUE/{} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate {} \
    --num_train_epochs {} \
    --save_steps 0 \
    --seed {} \
    --output_dir ./saved_models/{}/{}/{} \
    --overwrite_cache \
    --overwrite_output_dir\
    --log_id {}""",

    "train_highway":
        r"""python -um examples.run_highway_glue \
    --model_type {} \
    --model_name_or_path {} \
    --task_name {} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /h/xinji/projects/GLUE/{} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=8 \
    --learning_rate {} \
    --num_train_epochs {} \
    --overwrite_output_dir \
    --seed {} \
    --output_dir ./saved_models/{}/{}/{} \
    --save_steps 0 \
    --overwrite_cache \
    --train_routine {}\
    --log_id {}""",

    "eval_highway":
        r"""python -um examples.run_highway_glue \
    --model_type {} \
    --model_name_or_path ./saved_models/{}/{}/{} \
    --task_name {} \
    --do_eval \
    --do_lower_case \
    --data_dir /h/xinji/projects/GLUE/{} \
    --output_dir ./saved_models/{}/{}/{} \
    --max_seq_length 128 \
    --seed {} \
    --eval_each_highway \
    --early_exit_entropy {} \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1\
    --train_routine {}\
    --log_id {}"""
}

if flavor == "eval_highway":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model,
        dataset,
        routine + '-' + seed,
        dataset,
        dataset,
        model,
        dataset,
        routine + '-' + seed,
        seed,
        entropy,
        routine,
        filecount
    )
elif flavor == "train_highway":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model + get_model_suffix(model),
        dataset,
        dataset,
        best_learning_rate[model][dataset],
        best_epochs[model],
        seed,
        model,
        dataset,
        routine + '-' + seed,
        routine,
        filecount
    )
    if HP is not None:
        groups = map(lambda x: x.split(','), HP.split(';'))
        for g in groups:
            script += ' --' + g[0] + ' ' + g[1]
elif flavor == "raw":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model + get_model_suffix(model),
        dataset,
        dataset,
        best_learning_rate[model][dataset],
        best_epochs[model],
        seed,
        model,
        dataset,
        flavor + '-' + seed,
        filecount
    )
elif flavor == "entropy":
    entropies = entropy_selection[model][dataset].split(',')
    print(entropies)
    for curr_entropy in entropies:
        script = script_template["eval_highway"].format(
            model[:model.index('-')],
            model,
            dataset,
            routine + '-' + seed,
            dataset,
            dataset,
            model,
            dataset,
            routine + '-' + seed,
            seed,
            curr_entropy,
            routine,
            filecount
        )
        script = script + "\\\n    --no_comet"
        if HP=='testset':
            script = script + "\\\n    --testset"
        if inter == 'True':
            inter_run(script)
        else:
            submit_run(script, filecount)
        print("\n")
    exit(0)
elif flavor == 'limit':

    experiment = cm.Experiment(project_name='highway',
                               log_code=False,
                               auto_output_logging=False,
                               parse_args=False,
                               auto_metric_logging=False)
    experiment.log_parameters({
        "log_id": filecount,
        "model_and_size": model,
        "train_routine": "limit",
        "task_name": dataset
    })

    num_layers = 12 if 'base' in model else 24

    each_layer_fname = './plotting/saved_models/{}/{}/limit-42/each_layer.npy'.format(model, dataset)
    if os.path.exists(each_layer_fname):
        each_layer_result = list(np.load(each_layer_fname))
        i_start = len(each_layer_result)
    else:
        each_layer_result = []
        i_start = 0
        if not os.path.exists(os.path.dirname(each_layer_fname)):
            os.makedirs(os.path.dirname(each_layer_fname))

    for i in range(i_start, num_layers-1):
    # for i in range(1):
        script = script_template['train_highway'].format(
            model[:model.index('-')],
            model + get_model_suffix(model),
            dataset,
            dataset,
            best_learning_rate[model][dataset],
            best_epochs[model],
            seed,
            model,
            dataset,
            routine + '-' + seed,
            routine,
            filecount if i==i_start else "debug"
        )
        script = script + "\\\n    --no_comet\\\n    --limit_layer {}".format(i)
        print(script)

        target_file = './saved_models/{}/{}/{}/limit.npy'.format(model, dataset, "limit-42")
        if not os.path.exists(target_file):
            start_time = -1
        else:
            start_time = os.path.getmtime(target_file)

        if inter == "True":
            inter_run(script)
        else:
            submit_run(script, filecount)

        while True:
            time.sleep(10)
            if os.path.exists(target_file) and os.path.getmtime(target_file)>start_time:
                break

        npload = np.load(target_file)
        each_layer_result.append(npload[0])
        np.save(each_layer_fname, np.array(each_layer_result))
    npload = np.load('./plotting/saved_models/{}/{}/{}/each_layer.npy'.format(model, dataset, "two_stage-42"))
    each_layer_result.append(npload[-1])
    np.save(each_layer_fname, np.array(each_layer_result))

    experiment.log_metric("final result", each_layer_result[-1])
    experiment.log_other(
        "Each layer result",
        ' '.join([str(int(100 * x)) for x in each_layer_result]))

    exit(0)
else:
    print("Wrong flavor")
    exit(1)



print(script)
if inter == "True":
    inter_run(script)
else:
    submit_run(script, filecount)
