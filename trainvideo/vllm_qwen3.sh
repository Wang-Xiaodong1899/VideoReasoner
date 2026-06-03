CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/bn/wxd-video-understanding/wangxd/models/Qwen3-8B \
    --tokenizer /mnt/bn/wxd-video-understanding/wangxd/models/Qwen3-8B \
    --dtype bfloat16 \
    --port 8002 \
    --gpu-memory-utilization 0.7
# CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/bn/wxd-video-understanding/wangxd/models/Qwen3-8B