import sys
import os
import importlib.util
import sys
import os

# transformers_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/transformers/src/transformers"
# spec = importlib.util.spec_from_file_location("transformers", os.path.join(transformers_path, "__init__.py"))
# transformers = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(transformers)

# print(transformers.__file__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")


from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re

# from perception_encoder.keyframe_api import process

from qwen_vl_utils import process_vision_keyframes_info

# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


# model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-llava-178k-SFT/Qwen2.5-VL-7B-Instruct-llava-178k-SFT-Video-Keyframes-Max300-F128"
print(f"eval {model_path}")

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Keyframe-Prompt-Reason/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-TGRPO-Edit-fix-time-step300-Keyframe-Prompt-Reasoning-GQA_Charade_ActivityNet-7k-0713-N1-1e-6/checkpoint-40"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

tokenizer = processor.tokenizer

# x = processor.tokenizer.encode("<|im_start|>assistant\n\n")
# import pdb; pdb.set_trace()

existing_specials = processor.tokenizer.special_tokens_map.get("additional_special_tokens", [])
print(f"current special tokens: {existing_specials}")

seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>

print(f"seg_token_id: {seg_token_id}")

tool_token_id = processor.tokenizer.convert_tokens_to_ids("<|keyframe_selection_tool|>") # <|segment_pad|>

print(f"<|keyframe_selection_tool|>: {seg_token_id}")

keyframes_embed_id = processor.tokenizer.convert_tokens_to_ids("<|keyframes_embed|>") # <|segment_pad|>

print(f"<|keyframes_embed|>: {keyframes_embed_id}")

query = """
What does the person do with the white powdery substance from the larger bowl?
A. They sprinkle it over the countertop
B. They pour it into the food processor
C. They mix it with a liquid
D. They use a spoon to add it to the food processor
"""

query = """
What is the number of the first lipstick she used?"
A. 600, B. 656, C. 866, D. 999
"""

# query = """
# What kind of hat does the little boy wear?
# A. Cowboy hat B. Beret C. Baseball cap D. Top hat
# """
# /mnt/bn/wk-data-storage/wuzhirong/datasets/LVBench/all_videos/-hgaSElC3wU.mp4

query = """
How many red socks are above the fireplace at the end of this video?
A. 1 B. 4 C. 2 D. 3
"""

VIDEO_QUESTION_TEMPLATE = (
    "Give you a video. Please think carefully about your reasoning strategy before answering the question.\n"
    "There are three possible reasoning modes:\n"
    "1. **Event localization**: If understanding the video requires identifying a key event (e.g., due to unclear or complex dynamics), first locate the event using the format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where the ratios are floats between 0 and 1 indicating the relative position in the video.\n"
    "   You can then use <|video_zoomin|> to inspect the event content more closely.\n"
    "2. **Keyframe selection**: If identifying specific visual moments is more helpful, use the keyframe selection tool by inserting <|keyframe_selection_tool|> and provide the result in the format: <|keyframe_start|>[list_of_frame_indices]<|keyframe_end|>. You can then reason based on the visual content using <|keyframes_embed|>.\n"
    "3. **Direct reasoning**: If the video is already clear enough, you may proceed directly without locating events or selecting keyframes.\n"
    "Please begin with your chosen reasoning path and present your thought process and final answer using the following tags:\n"
    "<think> your reasoning process here </think> <answer> your answer here </answer>\n"
    "Question: {Question}\n"
)

system_message = "You are a very smart multimodal assistant that can understand videos. "

QUESTION_TEMPLATE = VIDEO_QUESTION_TEMPLATE

messages = [
    {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4",
                "max_pixels": 360 * 420,
                "fps": 2.0,
            },
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=query)},
        ]
    },
]

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True # NOTE: False to drop "<|im_start|>assistant"
# )

# keyframe_indices_dict_list = process(query, "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4")

# print(f"keyframe_indices_dict_list: {keyframe_indices_dict_list}")

# keyframe_indices_list = []
# for item in keyframe_indices_dict_list:
#     keyframe_indices_list.append(item["key_index"])

# keyframe_indices_list = list(set(keyframe_indices_list))

keyframe_indices_list = [114]

print("---------Third Input------------------------------")

keyframe_pad_str = "<|vision_start|><|image_pad|><|vision_end|>" * len(keyframe_indices_list)

prefix = f"""
<|im_start|>system
You are a very smart multimodal assistant that can understand videos. 
<|im_end|>
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>
Give you a video. Please think carefully about your reasoning strategy before answering the question.
There are three possible reasoning modes:
1. **Event localization**: If understanding the video requires identifying a key event (e.g., due to unclear or complex dynamics), first locate the event using the format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where the ratios are floats between 0 and 1 indicating the relative position in the video.
   You can then use <|video_zoomin|> to inspect the event content more closely.
2. **Keyframe selection**: If identifying specific visual moments is more helpful, use the keyframe selection tool by inserting <|keyframe_selection_tool|> and provide the result in the format: <|keyframe_start|>[list_of_frame_indices]<|keyframe_end|>. You can then reason based on the visual content using <|keyframes_embed|>.
3. **Direct reasoning**: If the video is already clear enough, you may proceed directly without locating events or selecting keyframes.
Please begin with your chosen reasoning path and present your thought process and final answer using the following tags:
<think> your reasoning process here </think> <answer> your answer here </answer>
Question: {query}

<|im_end|>
<|im_start|>assistant
<think> I want to use the keyframe selection tool <|keyframe_selection_tool|> to identify the relevant frames, and the selection result is <|keyframe_start|>{keyframe_indices_list}<|keyframe_end|>. By looking at the visual content of these keyframes <|keyframes_embed|>"""

prefix = prefix + keyframe_pad_str + ", I analyze the details provided for each frame.\n\n"

print(prefix)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4",
                "max_pixels": 768 * 768,
                "fps": 2.0,
            },
        ]
    },
]

image_inputs, video_inputs, keyframes_inputs, video_kwargs = process_vision_keyframes_info(messages, keyframe_indices_list, return_video_kwargs=True)

inputs = processor(
    images=keyframes_inputs,
    text=[prefix],
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True, top_p=1.0)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("--------Third Output---------------------------")

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text[0])

