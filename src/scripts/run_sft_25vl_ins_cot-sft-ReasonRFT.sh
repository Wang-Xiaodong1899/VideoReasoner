export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO 

cd src/r1-v

RANK=${1}

export SWANLAB_PROJECT=Qwen2.5-VL-CoT-SFT
export SWANLAB_NAME=Qwen2.5-VL-7B-Instruct-zero3-offload-ReasonRFT_0830

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SWANLAB_NAME}.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_cot_think.py \
    --output_dir /mnt/bn/wxd-video-understanding/wangxd/ckpt/${SWANLAB_PROJECT}/${SWANLAB_NAME} \
    --model_name_or_path /mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name /mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/VideoRFT/VideoRFT-Video-CoT-Data_15840.json \
    --deepspeed local_scripts/zero3_offload.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to swanlab \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name ${SWANLAB_NAME} \
    --save_steps 100 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 10 \