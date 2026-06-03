CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 400 --end 450 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 450 --end 500 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 500 --end 550 &
CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 550 --end 600 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 600 --end 650 &
CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 650 --end 700 &
CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 700 --end 750 &
CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 750 --end 760 &
wait
echo "All Inference Done!!!!!!!!!!"