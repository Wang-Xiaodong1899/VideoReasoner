CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 0 --end 30 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 30 --end 60 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 60 --end 90 &
CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 90 --end 120 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 120 --end 150 &
CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 150 --end 180 &
CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 180 --end 210 &
CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 210 --end 240 &
wait
echo "All Inference Done!!!!!!!!!!"
# 465 videos