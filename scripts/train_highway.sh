#!/bin/bash

#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=15:0:0
#SBATCH --output=/dev/null

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/home/xinji/scratch/GLUE

MODEL_TYPE=${1}
MODEL_SIZE=${2}
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
if [ $MODEL_TYPE = 'distilbert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MODEL_TYPE = 'albert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-v2
fi


if [ -z $SLURM_SRUN_COMM_HOST ]
then
  # non-interactive
  exec &> ${SLURM_TMPDIR}/slurm_out
fi

LR=2e-5
if [[ $ROUTINE = *-lte ]]
then
  EPOCHS=1
  LR=2e-4
fi


echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
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
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --overwrite_output_dir \
  --seed $SEED \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED} \
  --plot_data_dir ./plotting/ \
  --save_steps 0 \
  --overwrite_cache \
  --train_routine $ROUTINE \
  --log_id $SLURM_JOB_ID

if [ -z $SLURM_SRUN_COMM_HOST ]
then
  cp ${SLURM_TMPDIR}/slurm_out ./logs/${SLURM_JOB_ID}.slurm_out
fi
