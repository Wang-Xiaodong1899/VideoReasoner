from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re

# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-yes-1k7-ep3-fix-fps-videoP128F48"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-yes-1k7-ep10-fix-fps-videoP128F48"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-2k5-ep10-fix-fps-videoP128F48"

# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-Mix3k-ep5-videoP128F48"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-3k-ep10-videoP128F48/checkpoint-955/"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Video-R1-SFT-Video-Zoomin-3k-ep10-videoP128F48/checkpoint-955"
# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep10-videoP128F48-0624"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Charades-Activity-f64-sft3k-GRPO-clip-iou-max64-0701-Time-Prompt-Reasoning-0705-N1/checkpoint-100/"

print(f"eval {model_path}")
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Charades-Activity-f64-sft3k-GRPO-clip-iou-max64-0701")

tokenizer = processor.tokenizer

# x = processor.tokenizer.encode("<|im_start|>assistant\n\n")
# import pdb; pdb.set_trace()

existing_specials = tokenizer.special_tokens_map.get("additional_special_tokens", [])
print(f"current special tokens: {existing_specials}")

# new_specials = ["<|segment_pad|>", "<|event_start|>", "<|event_end|>", "<|video_zoomin|>"]

# all_specials = list(existing_specials) + new_specials

# print(f"latest special tokens: {all_specials}")

print(f"old vocab size: {len(tokenizer)}") 

# num_new_tokens = tokenizer.add_special_tokens({
#     "additional_special_tokens": all_specials
# })

# print(f"new tokens: {num_new_tokens}")

# print(f"New vocab size: {len(tokenizer)}")

seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>

print(f"seg_token_id: {seg_token_id}")

query = """
What does the person do with the white powdery substance from the larger bowl?
A. They sprinkle it over the countertop
B. They pour it into the food processor
C. They mix it with a liquid
D. They use a spoon to add it to the food processor
"""


VIDEO_QUESTION_TEMPLATE = (
    "Give you a video. Please think about your reasoning before answering the question.\n"
    "If the content of the question in the video is unclear or difficult to see, first locate the key event by generating the start and end times in the following format: <|event_start|> [t_start, t_end] <|event_end|>.\n"
    "Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
    "Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>\n"
    "Question: {Question}\n"
)

system_message = "You are a very smart multimodal assistant that can understand videos. "

QUESTION_TEMPLATE = VIDEO_QUESTION_TEMPLATE

query = "What number of lipstick is this girl using?"



prefix = f"""
<|im_start|>system
You are a very smart multimodal assistant that can understand videos. 
<|im_end|>
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>
Give you a video. Please think about your reasoning before answering the question.
If the content of the question in the video is unclear or difficult to see, first locate the key event by generating the start and end times in the following format: <|event_start|> [t_start, t_end] <|event_end|>.
Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>
Question:
{query}

<|im_end|>
<|im_start|>assistant
<think> To determine the number of lipstick she is using, I need to observe the specific lipstick product she applies during her makeup routine. The relevant event occurs between <|event_start|>[78.0, 92.3]<|event_end|>. Focusing on this segment <|video_zoomin|><|segment_pad|>,"""

query = "In the movie video, what did the monkey give to the protagonist?"
path = "/mnt/bn/wk-data-storage/wuzhirong/datasets/LVBench/all_videos/Za2Z_JRxCuk.mp4"

prefix = f"""
<|im_start|>system
You are a very smart multimodal assistant that can understand videos. 
<|im_end|>
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>
Give you a video. Please think about your reasoning before answering the question.
If the content of the question in the video is unclear or difficult to see, first locate the key event by generating the start and end times in the following format: <|event_start|> [t_start, t_end] <|event_end|>.
Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>
Question:
{query}

<|im_end|>
<|im_start|>assistant
<think> To determine what the monkey gave to the protagonist, I need to observe the specific interaction between the monkey and the protagonist where the gift is presented. The relevant event occurs between <|event_start|>[1070.0, 1090.0]<|event_end|>. Focusing on this timeframe <|video_zoomin|><|segment_pad|>,"""


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": path,
                # "max_pixels": 360 * 420,
                # "fps": 2.0,
                "fps": 4.0,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
                "total_pixels": 20480 * 28 * 28,
            },
        ]
    },
]

times = [1070.0, 1090.0]

print(prefix)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
_, segment_inputs = process_vision_given_multi_durations(messages, key_durations=times, return_video_kwargs=False, pad_frame=False, max_frames=32)
inputs = processor(
    text=[prefix],
    videos=video_inputs,
    segments=segment_inputs,
    fps=2, # NOTE ZOOM_FPS change here
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, top_p=1.0, use_cache=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
# whether skip the special token
print(output_text[0])
