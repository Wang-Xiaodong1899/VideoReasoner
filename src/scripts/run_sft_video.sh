cd src/r1-v

export WANDB_PROJECT=Video-R1-Reproduce
export WANDB_NAME=sft-8xH20-debug

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-7B-Video-7B-f32-cot-sft-debug" \
    --model_name_or_path /root/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name /mnt/bn/multimodal-datasets-hl/wangxd/data/Video-R1-data/Video-R1-COT-165k-filter.json \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-Video-cot-sft \
    --save_steps 5000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 2 \