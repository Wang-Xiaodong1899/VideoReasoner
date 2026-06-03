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

from perception_encoder.keyframe_api import process, process_queries

from qwen_vl_utils import process_vision_keyframes_info

path = "/mnt/bn/wk-data-storage/wuzhirong/datasets/LVBench/all_videos/Za2Z_JRxCuk.mp4"


print(process_queries(["monkey"], path))


# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


# model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-llava-178k-SFT/Qwen2.5-VL-7B-Instruct-llava-178k-SFT-Video-Keyframes-Max300-F128"
print(f"eval {model_path}")
# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
#     attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_path)

tokenizer = processor.tokenizer

# x = processor.tokenizer.encode("<|im_start|>assistant\n\n")
# import pdb; pdb.set_trace()

existing_specials = processor.tokenizer.special_tokens_map.get("additional_special_tokens", [])
print(f"current special tokens: {existing_specials}")

# new_specials = ["<|segment_pad|>", "<|event_start|>", "<|event_end|>", "<|video_zoomin|>",
#                 "<|keyframe_selection_tool|>", "<|keyframe_start|>", "<|keyframe_end|>",
#                 "<|keyframes_embed|>", "<|keyframes_pad|>"
#                 ]

# all_specials = list(existing_specials) + new_specials

# print(f"latest special tokens: {all_specials}")

# print(f"old vocab size: {len(processor.tokenizer)}") 

# num_new_tokens = processor.tokenizer.add_special_tokens({
#     "additional_special_tokens": all_specials
# })

# print(f"new tokens: {num_new_tokens}")

# print(f"New vocab size: {len(processor.tokenizer)}")

# new_specials = ["<|segment_pad|>", "<|event_start|>", "<|event_end|>", "<|video_zoomin|>"]

# all_specials = list(existing_specials) + new_specials

# print(f"latest special tokens: {all_specials}")

# print(f"old vocab size: {len(tokenizer)}") 

# num_new_tokens = tokenizer.add_special_tokens({
#     "additional_special_tokens": all_specials
# })

# print(f"new tokens: {num_new_tokens}")

# print(f"New vocab size: {len(tokenizer)}")

seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>

print(f"seg_token_id: {seg_token_id}")

tool_token_id = processor.tokenizer.convert_tokens_to_ids("<|keyframe_selection_tool|>") # <|segment_pad|>

print(f"<|keyframe_selection_tool|>: {seg_token_id}")

keyframes_embed_id = processor.tokenizer.convert_tokens_to_ids("<|keyframes_embed|>") # <|segment_pad|>

print(f"<|keyframes_embed|>: {keyframes_embed_id}")

# processor.chat_template = """{% set image_count = namespace(value=0) %}
# {% set video_count = namespace(value=0) %}
# {% for message in messages %}
#   {% if loop.first and message['role'] != 'system' %}
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
#   {% endif %}
# <|im_start|>{{ message['role'] }}
#   {% if message['content'] is string %}
# {{ message['content'] }}<|im_end|>
#   {% else %}
#     {% for content in message['content'] %}
#       {# 忽略 segment 类型的 content #}
#       {% if content['type'] == 'segment' %}
#         {# skip segment #}
#       {% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}
#         {% set image_count.value = image_count.value + 1 %}
#         {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
# <|vision_start|><|image_pad|><|vision_end|>
#       {% elif content['type'] == 'video' or 'video' in content %}
#         {% set video_count.value = video_count.value + 1 %}
#         {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
# <|vision_start|><|video_pad|><|vision_end|>
#       {% elif 'text' in content %}
# {{ content['text'] }}
#       {% endif %}
#     {% endfor %}
# <|im_end|>
#   {% endif %}
# {% endfor %}
# {% if add_generation_prompt %}
# <|im_start|>assistant
# {% endif %}

# """

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

# /mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4

# query = """
# Which task was not completed by the robots?
# A. Vault. B. Split. C. Balance beam. D. Backflip.
# """

# query = """
# How many glass discs are there inside the disassembled lens in the video, at minimum?
# A. 4. B. 5. C. 3. D. 2.
# """

# query = """
# What is the logo on the pitcher's chest who wears a blue and red sports shirt and orange helmet?
# A. A flower. B. A row of letters. C. A plane. D. A tick
# """

# query = """
# What is the video telling when the burger placed in the upper right corner at the end of the video first appears?
# A. Beef with spices came from Russia to Germany. 
# B. The steak began to be sandwiched between two pieces of bread.
# C. Steak burgers spread throughout the United States.
# D. The standardization of hamburgers.
# """

# query = """
# Which best summarizes the content of the video?
# A. Supply and demand.
# B. Bananas supply.
# C. Business competition.
# D. Banana selling.
# """

# VIDEO_QUESTION_TEMPLATE = (
#     "Give you a video. Please think about your reasoning before answering the question.\n"
#     "If the content of the question in the video is unclear or difficult to see, first locate the key event by generating the start and end times in the following format: <|event_start|> [t_start, t_end] <|event_end|>.\n"
#     "Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
#     "Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>\n"
#     "Question: {Question}\n"
# )
VIDEO_QUESTION_TEMPLATE = (
    "Give you a video. Please think about your reasoning before answering the question.\n"
    "If the question content in the video is unclear or difficult to see, first use the keyframe selection tool <|keyframe_selection_tool|> to locate the keyframe: <|keyframe_start|> [idx1, idx2,...] <|keyframe_end|>.\n"
    "Then, use the <|keyframes_embed|> tag to zoom in on the keyframe content for a closer look.\n"
    "Please show your reasoning and answer in the following tags: <think>Reasoning process</think> <answer>Answer here</answer>\n"
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

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True # NOTE: False to drop "<|im_start|>assistant"
)
# import pdb; pdb.set_trace()

# print(text)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs # fps in video_kwargs
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True, top_p=1.0)

print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')


generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

truncated_list = []
for tensor in generated_ids:
    mask = (tensor == tool_token_id)
    occurrences = torch.nonzero(mask).flatten()  # 所有151668的位置
    
    if len(occurrences) >= 2:
        stop_index = occurrences[1].item() + 1
        truncated_tensor = tensor[:stop_index]
    else:
        truncated_tensor = tensor
    
    truncated_list.append(truncated_tensor)
# print(truncated_list)

truncated_text = processor.batch_decode(
    truncated_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

# print(truncated_text[0])

# import pdb; pdb.set_trace()

# whether skip the special token

# truncated_text = [re.sub(r'(<\|video_pad\|>)+', r'<|video_pad|>', truncated_text[0])]
# print(truncated_text[0])
truncated_text = [truncated_text[0] + ", and the selection result is <|keyframe_start|>"]

keyframe_indices_dict_list = process(query, "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4")

print(f"keyframe_indices_dict_list: {keyframe_indices_dict_list}")

keyframe_indices_list = []
for item in keyframe_indices_dict_list:
    keyframe_indices_list.append(item["key_index"])

keyframe_indices_list = list(set(keyframe_indices_list))
# keyframe_indices_list = [0, 10, 115]

truncated_text = [re.sub(r'(<\|video_pad\|>)+', r'<|video_pad|>', truncated_text[0])]
# print(truncated_text[0])

truncated_text = [truncated_text[0] + str(keyframe_indices_list) + "<|keyframe_end|>."]

# print(f"-----------second input -------------------\ntruncated_text: {truncated_text[0]}")

# messages[1]["content"][0]["fps"] = 4 # NOTE enable fps in zoom_in video process
# print(messages)
# _, segment_inputs = process_vision_given_multi_durations(messages, key_durations=timeline)
inputs = processor(
    text=truncated_text,
    videos=video_inputs,
    fps=messages[1]["content"][0]["fps"], # NOTE ZOOM_FPS change here
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True, top_p=1.0)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("--------Second Output---------------------------")

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
# whether skip the special token
# print(output_text[0])

truncated_list = []
for tensor in generated_ids:
    mask = (tensor == keyframes_embed_id)
    occurrences = torch.nonzero(mask).flatten()  # 所有151668的位置
    
    if len(occurrences) >= 2:
        stop_index = occurrences[1].item() + 1
        truncated_tensor = tensor[:stop_index]
    else:
        truncated_tensor = tensor
    
    truncated_list.append(truncated_tensor)

truncated_text = processor.batch_decode(
    truncated_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

truncated_text = [re.sub(r'(<\|video_pad\|>)+', r'<|video_pad|>', truncated_text[0])]

# print(truncated_text[0])

print("---------Third Input------------------------------")

truncated_text = [truncated_text[0] + "<|keyframes_pad|>" * len(keyframe_indices_list)]
print(truncated_text[0])

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
    text=truncated_text,
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

