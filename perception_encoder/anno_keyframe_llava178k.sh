CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=64 python /mnt/bn/wxd/wangxd/VideoReasoner/perception_encoder/keyframe_api_llava178k.py --start 0 --end 400 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=64 python /mnt/bn/wxd/wangxd/VideoReasoner/perception_encoder/keyframe_api_llava178k.py --start 400 --end 800 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=64 python /mnt/bn/wxd/wangxd/VideoReasoner/perception_encoder/keyframe_api_llava178k.py --start 800 --end 1200 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=64 python /mnt/bn/wxd/wangxd/VideoReasoner/perception_encoder/keyframe_api_llava178k.py --start 1200 --end 1730 &