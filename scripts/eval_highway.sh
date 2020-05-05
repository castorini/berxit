#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p p100
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --output=logs/%j.slurm_out

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/h/xinji/projects/GLUE

MODEL_TYPE=${1}
MODEL_SIZE=${2}  # change partition to t4 if large
DATASET=${3}
SEED=42
ROUTINE=${4}

ENTROPY=0.1
LTE_TH=0 #"0.1,14;0.15,10"


echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
python -um examples.run_highway_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED} \
  --task_name $DATASET \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED} \
  --max_seq_length 128 \
  --seed $SEED \
  --eval_each_highway \
  --early_exit_entropy $ENTROPY \
  --eval_highway \
  --overwrite_cache \
  --per_gpu_eval_batch_size=1 \
  --train_routine $ROUTINE \
  --lte_th $LTE_TH \
  --log_id $SLURM_JOB_ID
