CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 240 --end 270 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 270 --end 300 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 300 --end 330 &
CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 330 --end 360 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 360 --end 390 &
CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 390 --end 420 &
CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 420 --end 450 &
CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 450 --end 480 &
wait
echo "All Inference Done!!!!!!!!!!"
# 465 videos