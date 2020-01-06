import sys
import os

flavor, model, dataset, seed, routine, entropy, inter = sys.argv[1:]
# flavor: raw, train_highway, eval_highway
# model: bert-base, bert-large, roberta-base, roberta-large
# dataset: MRPC, SST-2, RTE, QNLI
# seed: 42, 9102, 4396
# routine: raw(?), two_stage(ACL), all(new)
# entropy: anything, -1 will evaluate all layers

entropy_selection = {
    "bert-base": {
        "MRPC":  "0,0.05,0.1,0.15,0.2,0.3,0.4,0.5",
        "SST-2": "0,0.001,0.005,0.01,0.05,0.2,0.3,0.4",
        "QNLI":  "0,0.06,0.1,0.14,0.15,0.2,0.3,0.35",
        "RTE":   "0,0.2,0.25,0.3,0.35,0.4,0.6,0.65"
    },
    "roberta-base": {
        "MRPC":  "0,0.05,0.1,0.15,0.2,0.3,0.5,0.55,0.6",
        "SST-2": "0,0.001,0.01,0.1,0.2,0.3,0.5,0.55,0.6",
        "QNLI":  "0,0.06,0.1,0.2,0.25,0.45,0.5,0.55",
        "RTE":   "0,0.03,0.05,0.25,0.3,0.45,0.5"
    },
    "bert-large": {
        "SST-2": "0,0.001,0.005,0.01,0.05,0.2,0.3,0.4,0.5,0.55,0.6"
    },
    "roberta-large": {
        "MRPC":  "0,0.05,0.1,0.15,0.2,0.3,0.5,0.55,0.6",
        "SST-2": "0,0.001,0.01,0.1,0.2,0.3,0.5,0.55,0.6",
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
        }
}
best_epochs = {
    "bert-base": 3,
    "bert-large": 3,
    "roberta-base": 10,
    "roberta-large": 10
}

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
    --overwrite_output_dir""",

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
    --train_routine {}""",

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
    --eval_each_highway \
    --early_exit_entropy {} \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1\
    --train_routine {}"""
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
        entropy,
        routine
    )
elif flavor == "train_highway":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model + ("-uncased" if "bert-" in model else ""),
        dataset,
        dataset,
        best_learning_rate[model][dataset],
        best_epochs[model],
        seed,
        model,
        dataset,
        routine + '-' + seed,
        routine
    )
elif flavor == "raw":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model + ("-uncased" if "bert-" in model else ""),
        dataset,
        dataset,
        best_learning_rate[model][dataset],
        best_epochs[model],
        seed,
        model,
        dataset,
        flavor + '-' + seed
    )
elif flavor == "entropy":
    if inter != "True":
        print("Entropy must be interactive")
        exit(1)
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
            curr_entropy
        )
        # print(script)
        os.system(script)
        print("\n")
    exit(0)
else:
    print("Wrong flavor")
    exit(1)

print(script)
if inter == "True":
    # interactive node
    os.system(script)
else:
    # submit job
    with open("slurm_submit.sh", 'w') as f:
        print(script, file=f)
    os.system("python ~/submit.py slurm_submit.sh")
