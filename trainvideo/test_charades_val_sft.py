from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re
import json
import os
from tqdm import tqdm
import csv

# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/val.json", "r") as f:
#     val_data = json.load(f)

# keys = val_data.keys()

# # # # All keys
# keys = list(keys)
# # keys = list(keys)[:100]
# test_data = []

# for key in tqdm(keys):
#     vid = key
#     video_path = os.path.join("/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1", vid+".mp4")
#     video_info = val_data[vid]
#     timestamps = video_info["timestamps"]
#     sentences = video_info["sentences"]
#     duration = video_info["duration"]

#     question = f"""What was the scene like when "{sentences[0]}" happened?"""
#     test_data.append({
#         "video_path": video_path,
#         "question": question,
#         "timestamp": timestamps[0],
#         "duration": duration,
#     })
    
# #     question = f"""
# # Give the query: "{sentences[0]}", when does the described content occur in the video? Use ‘ss.ff’ as time format.
# #     """
    # for idx, timestamp in enumerate(timestamps):
    #     sentence = sentences[idx]
    #     # question = f"""Find start and end seconds for: "{sentence}", please return the start and end seconds."""
    #     question = f"""What was the scene like when "{sentence}" happened?"""
    #     # question = f"""Give the query: "{sentence}", when does the described content occur in the video? Use ‘ss.ff’ as time format."""
    #     test_data.append({
    #         "video_path": video_path,
    #         "question": question,
    #         "timestamp": timestamp,
    #         "duration": duration,
    #     })

# save test_data to a csv
# with open("/mnt/bn/wxd-video-understanding/wangxd/repo/wzr_eval/xiaodong/scripts/test_charades_val_all_for_seed15vl.csv", "w") as f:
#     fieldnames = ["video_path", "question", "timestamp"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     for data in test_data:
#         writer.writerow(data)

# save test_data to a json file
# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/test_charades_val_all_for_ours_sft.json", "w") as f:
#     json.dump(test_data, f, indent=4)

# import pdb; pdb.set_trace()


# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct

# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct"
# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Charades-Activity-f64-sft3k-GRPO-clip-iou-max64-0701"
model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep5-videoP80F64-N2-percentage-0706"
print(f"eval {model_path}")
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



VIDEO_QUESTION_TEMPLATE = (
    "Give you a video. Please think about your reasoning before answering the question.\n"
    "If the content of the question in the video is unclear or difficult to see, first locate the key event by generating its location as a proportion of the video duration in the following format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where start_ratio and end_ratio are floats between 0 and 1, indicating the relative start and end points of the key event.\n"
    "Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
    "Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>\n"
    "Question: {Question}\n"
)

system_message = "You are a very smart multimodal assistant that can understand videos. "

QUESTION_TEMPLATE = VIDEO_QUESTION_TEMPLATE

def process(query, video_path):
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
                    "video": video_path,
                    "max_pixels": 768 * 768,
                    "fps": 2.0,
                },
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=query)},
            ]
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True # NOTE: False to drop "<|im_start|>assistant"
    )

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
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, top_p=1.0, use_cache=True)

    print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # print(generated_ids_trimmed)

    # clamp the generated_ids, take previous, if meet 151668
    truncated_list = []
    for tensor in generated_ids:
        mask = (tensor == 151668)
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
    # whether skip the special token

    truncated_text = [re.sub(r'(<\|video_pad\|>)+', r'<|video_pad|>', truncated_text[0])]
    # print(truncated_text[0])

    match = re.search(r'<\|event_start\|>\[(.*?)\]<\|event_end\|>', truncated_text[0])

    if match:
        list_str = match.group(1)  # 获取 "10.6, 17.4"
        timeline = [float(num) for num in list_str.split(', ')]
        print(timeline)
    else:
        print("未找到符合标签要求的内容")
        timeline = None

    return timeline

# reverse test_data
# test_data = test_data[::-1]
with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/test_charades_val_all_for_ours_sft.json", "r") as f:
    test_data = json.load(f)

our_pred = []
with open("our_sft_pred_all.jsonl", "w") as f:
    for item in tqdm(test_data):
        video_path = item['video_path']
        query = item['question']
        timeline = process(query, video_path)
        if timeline is None:
            continue
        duration = item['duration']
        timeline = [float(time) * duration for time in timeline]
        gt_time = item['timestamp']
        our_pred.append({
            'video_path': video_path,
            'query': query,
            'timeline': timeline,
            'gt_timeline': gt_time,
        })
        # save to jsonl file
        json.dump(our_pred[-1], f)
        f.write('\n')
        f.flush()


# save our_pred to json
with open('our_sft_pred_all.json', 'w') as f:
    json.dump(our_pred, f, indent=4)
