cd src/r1-v

export WANDB_PROJECT=Video-R1-Reproduce
export WANDB_NAME=Qwen2-VL-7B-base-sft-8xH20

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2-VL-7B-base-Video-7B-f32-cot-sft" \
    --model_name_or_path /mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2-VL-7B \
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
    --run_name Qwen2-VL-7B-Base-Video-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 2 \