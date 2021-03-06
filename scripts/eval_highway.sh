#!/bin/bash

#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=6:0:0
#SBATCH --output=/dev/null

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/home/xinji/scratch/GLUE

MODEL_TYPE=${1}
MODEL_SIZE=${2}
DATASET=${3}
SEED=42
ROUTINE=${4}

if [ ! -z ${5} ]
then
  NOCOMET_SWITCH='--no_comet'
  if [ ${5} = 'testset' ]
  then
    LTE_TH='-1'  # set it to "-1" to trigger eval_each_highway
                 # set it to "0.0" to trigger uncertainty recording
    TESTSET_SWITCH='--testset'
  else
    LTE_TH=${5}
    TESTSET_SWITCH=''
  fi
else
  LTE_TH='-1'
  NOCOMET_SWITCH=''
  TESTSET_SWITCH=''
fi


if [ -z $SLURM_SRUN_COMM_HOST ]
then
  # non-interactive
  exec &> ${SLURM_TMPDIR}/slurm_out
fi

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
  --eval_highway \
  --overwrite_cache \
  --per_gpu_eval_batch_size=1 \
  --train_routine $ROUTINE \
  --lte_th $LTE_TH \
  --log_id $SLURM_JOB_ID \
  $NOCOMET_SWITCH \
  $TESTSET_SWITCH


if [ -z $SLURM_SRUN_COMM_HOST ]
then
  cp ${SLURM_TMPDIR}/slurm_out ./logs/${SLURM_JOB_ID}.slurm_out
fi
