#!/bin/bash

#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=12:0:0
#SBATCH --output=/dev/null

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/home/xinji/scratch/GLUE

MODEL_TYPE=${1}
MODEL_SIZE=${2}
DATASET=${3}
SEED=42
ROUTINE=${4}
TESTSET=${5}  # generate result for the test set

TESTSET_SWITCH=''
if [ ! -z $TESTSET ] && [ $TESTSET = 'testset' ]
then
  TESTSET_SWITCH='--testset'
fi

if [ -z $SLURM_SRUN_COMM_HOST ]
then
  # non-interactive
  exec &> ${SLURM_TMPDIR}/slurm_out
fi

ENTROPIES="0.0"

echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE

for ENTROPY in $ENTROPIES; do
  printf "\nEntropy $ENTROPY\n"
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
    --early_exit_entropy $ENTROPY \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1 \
    --train_routine $ROUTINE \
    --log_id $SLURM_JOB_ID \
    --no_comet \
    $TESTSET_SWITCH

done


if [ -z $SLURM_SRUN_COMM_HOST ]
then
  cp ${SLURM_TMPDIR}/slurm_out ./logs/${SLURM_JOB_ID}.slurm_out
fi
