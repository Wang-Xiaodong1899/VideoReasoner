CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 280 --end 315 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 315 --end 350 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 350 --end 385 &
CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 385 --end 420 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 420 --end 455 &
CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 455 --end 490 &
CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 490 --end 525 &
CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 525 --end 560 &
wait
echo "All Inference Done!!!!!!!!!!"
# 551 videos
