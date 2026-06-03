export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO 

cd src/r1-v

RANK=${1}

export DEBUG_MODE="true"
FPS_MAX_FRAMES=80

export SWANLAB_PROJECT=Qwen25-SFT-Time-GRPO
export SWANLAB_NAME=Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-zero3-offload-0711-N3-TGRPO-Edit-fix-time

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SWANLAB_NAME}.txt"

# QWEN_PATH=/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-zero3-offload-0711-N2
QWEN_PATH=/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-zero3-offload-0712-N2
HF_DATASET=/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/charades_v1_train_grpo_iou_only.json

OUTPUT_DIR=/mnt/bn/wxd-video-understanding/wangxd/ckpt/${SWANLAB_PROJECT}/${SWANLAB_NAME}
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="local_scripts/zero3_offload.json"  


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="7" \
    --nnodes="3" \
    --node_rank=${RANK} \
    --master_addr="[2605:340:cd51:4900:a12f:4426:d1f3:3e2]" \
    --master_port="12352" \
    src/open_r1/grpo_clip_tgrpo_edit_reward.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-6 \
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
    --save_total_limit 2 \
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
    --vllm_gpu_memory_utilization 0.5 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
