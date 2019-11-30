import sys
import os

flavor, model, dataset, seed, entropy, inter = sys.argv[1:]
# flavor: raw, highway, eval_highway
# model: bert-base, bert-large, roberta-base, roberta-large
# dataset: MRPC, SST-2, RTE, QNLI
# seed: 42, 9102, 4396

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
    --data_dir /scratch/gobi1/xinji/GLUE/{} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate {} \
    --num_train_epochs {} \
    --save_steps 0 \
    --seed {} \
    --output_dir ./saved_models/{}/{}/{} \
    --overwrite_output_dir""",

    "highway":
        r"""python -um examples.run_highway_glue \
    --model_type {} \
    --model_name_or_path {} \
    --task_name {} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /scratch/gobi1/xinji/GLUE/{} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=8 \
    --learning_rate {} \
    --num_train_epochs {} \
    --overwrite_output_dir \
    --seed {} \
    --output_dir ./saved_models/{}/{}/{} \
    --save_steps 0 \
    --eval_after_first_stage""",

    "eval_highway":
        r"""python -um examples.run_highway_glue \
    --model_type {} \
    --model_name_or_path ./saved_models/{}/{}/{} \
    --task_name {} \
    --do_eval \
    --do_lower_case \
    --data_dir /scratch/gobi1/xinji/GLUE/{} \
    --output_dir ./saved_models/{}/{}/{} \
    --max_seq_length 128 \
    --eval_each_highway \
    --early_exit_entropy {} \
    --per_gpu_eval_batch_size=1"""
}

if flavor == "eval_highway":
    script = script_template[flavor].format(
        model[:model.index('-')],
        model,
        dataset,
        'highway-' + seed,
        dataset,
        dataset,
        model,
        dataset,
        'highway-' + seed,
        entropy
    )
elif flavor == "highway":
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
elif flavor.startswith("entropy"):
    if inter != "True":
        print("Entropy must be interactive")
        exit(1)
    entropies = flavor[flavor.index(':')+1:].split(',')
    for curr_entropy in entropies:
        script = script_template["eval_highway"].format(
            model[:model.index('-')],
            model,
            dataset,
            'highway-' + seed,
            dataset,
            dataset,
            model,
            dataset,
            'highway-' + seed,
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
