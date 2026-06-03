#!/bin/bash
export SWANLAB_PROJECT=Qwen25-SFT-Keyitem-GRPO
# export WANDB_NAME=sft-8xH20

cd src/r1-v

RANK=${1}

export DEBUG_MODE="true"
FPS_MAX_FRAMES=80

# export FPS_MAX_FRAMES=${FPS_MAX_FRAMES}

RUN_NAME=Qwen2.5-VL-7B-Instruct-f${FPS_MAX_FRAMES}-TGRPO-step500-GRPO-keyitems-max80-0721-N1-1e-6

export LOG_PATH=./${RUN_NAME}.txt


QWEN_PATH=/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2.5-VL-7B-Instruct-f80-GRPO-clip-iou-charades-9k-max80-ratio-0721-N2/checkpoint-550
HF_DATASET=/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/mix_keyframes_Charades_llava_GQA_data_new_0721.json


OUTPUT_DIR=/mnt/bn/wxd-video-understanding/wangxd/ckpt/${SWANLAB_PROJECT}/${RUN_NAME}
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="local_scripts/zero3_offload.json"  

# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank=${RANK} \
    --master_port="12352" \
    src/open_r1/grpo_clip_entity_grpo_0720.py \
    --use_vllm false \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_total_limit 10 \
    --save_only_model false \
    --temporal false \
    --len_control false \
    --use_std true \
    --use_length_norm true \
    --report_to swanlab \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
