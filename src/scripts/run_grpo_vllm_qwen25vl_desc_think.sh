#!/bin/bash
export WANDB_PROJECT=Video-Perception
# export WANDB_NAME=sft-8xH20

cd src/r1-v

export DEBUG_MODE="true"
FPS_MAX_FRAMES=32

export FPS_MAX_FRAMES=${FPS_MAX_FRAMES}

RUN_NAME=Qwen2.5-VL-7B-Instruct-Video-R1-f${FPS_MAX_FRAMES}-desc-new-template-think-video-GRPO-MAX196-maxlen1024-14k-0507

export LOG_PATH=./${RUN_NAME}.txt

# QWEN_PATH=/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct
QWEN_PATH=/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Video-R1-f32-desc-sup-BERT-new-template-video-GRPO-MAX196-maxlen1024-14k-0507/checkpoint-400
# HF_DATASET=/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter.json
# HF_DATASET=/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video.json # only video data
# HF_DATASET=/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video-desc-32b-830.json # supvision caption data
# HF_DATASET=/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video-desc-32b-merge-0504-14k.json
HF_DATASET=/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video-14k.json
OUTPUT_DIR=/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/${RUN_NAME}
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="local_scripts/zero3.json"  

# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12350" \
    src/open_r1/grpo_desc_sup_reason.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
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
    --save_steps 200 \
    --save_total_limit 1 \
    --save_only_model false \
    --temporal false \
    --len_control false \
    --use_std true \
    --use_length_norm true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
