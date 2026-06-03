#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO 

export SWANLAB_PROJECT=Qwen25-SFT-Time-Prompt-Reason

cd src/r1-v

RANK=${1}

export DEBUG_MODE="true"
FPS_MAX_FRAMES=64

# export FPS_MAX_FRAMES=${FPS_MAX_FRAMES}

export SWANLAB_NAME=Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-data-TGRPO-Edit-fix-time-best400-Time-Prompt-Reasoning-GQA_Charade_ActivityNet-7k-clipmaxf16-len-prompt-0714-N2

export LOG_PATH=./${SWANLAB_NAME}.txt


# QWEN_PATH=/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-zero3-offload-0711-N3-TGRPO-Edit-fix-time-back/checkpoint-250
QWEN_PATH=/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-zero3-offload-0711-N3-TGRPO-Edit-fix-time/checkpoint-400
# HF_DATASET=/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/stage_1_IoU_RLModel_pred_GT_time_ratio.json
HF_DATASET=/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/RL_GQA_Charade_ActivityNet_7k.json


OUTPUT_DIR=/mnt/bn/wxd-video-understanding/wangxd/ckpt/${SWANLAB_PROJECT}/${SWANLAB_NAME}
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="local_scripts/zero3_offload.json"  

# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="2" \
    --node_rank=${RANK} \
    --master_addr="[2605:340:cd51:4900:b726:1043:d242:b653]" \
    --master_port="12352" \
    src/open_r1/grpo_mix_time_prompt.py \
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
    --run_name ${SWANLAB_NAME} \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model false \
    --temporal false \
    --len_control true \
    --use_std true \
    --use_length_norm true \
    --report_to swanlab \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:0" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
