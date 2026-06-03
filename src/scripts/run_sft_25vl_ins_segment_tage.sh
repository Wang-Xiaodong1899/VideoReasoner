cd src/r1-v

RANK=${1}

export WANDB_PROJECT=Qwen2.5-VL-Finetune-Zoomin-SFT
export WANDB_NAME=Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep5-videoP80F64-N2-percentage-0706

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${WANDB_NAME}.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="2" \
    --node_rank=${RANK} \
    --master_addr="[2605:340:cd51:4900:b726:1043:d242:b653]" \
    --master_port="12354" \
    src/open_r1/sft_clip_tage.py \
    --output_dir /mnt/bn/wxd-video-understanding/wangxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME} \
    --model_name_or_path /mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name /mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_3k_query_time_0705_check.json \
    --deepspeed local_scripts/zero3_offload.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to swanlab \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name ${WANDB_NAME} \
    --save_steps 100 \
    --max_grad_norm 5 \
    --save_only_model true \
    --save_total_limit 2 \