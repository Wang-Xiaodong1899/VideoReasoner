# CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 0 --end 40 &
# CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 40 --end 70 &
# CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 70 --end 100 &
# CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 100 --end 130 &
# CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 130 --end 160 &
# CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 160 --end 190 &
# CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 190 --end 220 &
# CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 220 --end 250 &
# CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 250 --end 260 &
# CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 260 --end 270 &
# CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 270 --end 280 &
# CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 280 --end 290 &
# CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries.py --start 290 --end 300 &

# wait
CUDA_VISIBLE_DEVICES=0 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 0 --end 33 &
CUDA_VISIBLE_DEVICES=1 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 33 --end 66 &
CUDA_VISIBLE_DEVICES=2 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 66 --end 99 &
CUDA_VISIBLE_DEVICES=3 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 99 --end 132 &
CUDA_VISIBLE_DEVICES=4 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 132 --end 165 &
CUDA_VISIBLE_DEVICES=5 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 165 --end 198 &
CUDA_VISIBLE_DEVICES=6 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 198 --end 231 &
CUDA_VISIBLE_DEVICES=7 FPS_MAX_FRAMES=768 python trainvideo/get_mme_queries_supply.py --start 231 --end 264 &
wait
echo "All Inference Done!!!!!!!!!!"