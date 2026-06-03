export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO 

cd src/r1-v

RANK=${1}

export SWANLAB_PROJECT=Qwen2.5-VL-Finetune-Zoomin-SFT
export SWANLAB_NAME=Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep2-videoP80F64-percentage-no-event-continue-event-ep5-N1-0709-zero3-offload

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SWANLAB_NAME}.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank=${RANK} \
    --master_port="12354" \
    src/open_r1/sft_clip_tage_mix.py \
    --output_dir /mnt/bn/wxd-video-understanding/wangxd/ckpt/${SWANLAB_PROJECT}/${SWANLAB_NAME} \
    --model_name_or_path /mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep2-videoP80F64-percentage-no-event-N1-0709 \
    --dataset_name /mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_event_id_0_1k_query_time_latest_update_0708.json \
    --deepspeed local_scripts/zero3_offload.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to swanlab \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name ${SWANLAB_NAME} \
    --save_steps 50 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 2 \