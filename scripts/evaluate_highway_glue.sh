python -um examples.run_highway_glue \
    --model_type bert \
    --model_name_or_path ./saved_models/B3H3 \
    --task_name MRPC \
    --do_eval \
    --do_lower_case \
    --data_dir /scratch/gobi1/xinji/GLUE/MRPC \
    --output_dir ./saved_models/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1
