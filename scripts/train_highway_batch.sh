#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p p100
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --output=logs/%j.slurm_out

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/home/xinji/scratch/GLUE

MODEL_TYPE=${1}
MODEL_SIZE=${2}  # change partition to t4 if large
DATASET=${3}
SEED=42
ROUTINE=${4}

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
EPOCHS=10
if [ $MODEL_TYPE = 'bert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MODEL_TYPE = 'albert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-v2
fi

echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET

ALPHAS=(0.1 0.01)
BETAS=(3.0 3.0)

for i in ${!ALPHAS[@]}; do
  printf "\n$i ${ALPHAS[$i]} ${BETAS[$i]}\n"
  python -um examples.run_highway_glue \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $DATASET \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=8 \
    --learning_rate 2e-5 \
    --num_train_epochs $EPOCHS \
    --overwrite_output_dir \
    --seed $SEED \
    --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED} \
    --plot_data_dir ./plotting/ \
    --save_steps 0 \
    --overwrite_cache \
    --train_routine $ROUTINE \
    --alpha ${ALPHAS[$i]} \
    --beta ${BETAS[$i]} \
    --log_id $SLURM_JOB_ID

done
