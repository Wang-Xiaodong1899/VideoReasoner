cd src/r1-v

export WANDB_PROJECT=Qwen2-VL-Finetune-Video-R1-SFT
export WANDB_NAME=Qwen2-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${WANDB_NAME}.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_clip.py \
    --output_dir /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/${WANDB_NAME} \
    --model_name_or_path /mnt/bn/wxd-video-understanding/wangxd/models/Qwen2-VL-7B-Instruct \
    --dataset_name /mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_exist.json \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name ${WANDB_NAME} \
    --save_steps 50 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 1 \