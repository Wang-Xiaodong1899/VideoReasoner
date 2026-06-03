FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=0 python trainvideo/stage_1_grpo_model_generate_reason.py 4000 4100 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=1 python trainvideo/stage_1_grpo_model_generate_reason.py 4100 4200 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=2 python trainvideo/stage_1_grpo_model_generate_reason.py 4200 4300 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=3 python trainvideo/stage_1_grpo_model_generate_reason.py 4300 4400 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=4 python trainvideo/stage_1_grpo_model_generate_reason.py 4400 4500 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=5 python trainvideo/stage_1_grpo_model_generate_reason.py 4500 4600 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=6 python trainvideo/stage_1_grpo_model_generate_reason.py 4600 4700 &
FPS_MAX_FRAMES=64 CUDA_VISIBLE_DEVICES=7 python trainvideo/stage_1_grpo_model_generate_reason.py 4700 4800 &
wait
