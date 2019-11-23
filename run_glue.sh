export TASK_NAME=STS-B
export OUTPUT_DIR="./saved_models/"

python -um examples.run_glue \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR \
    --save_steps 719
