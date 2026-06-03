import json
from tqdm import tqdm
import pandas as pd
import os

from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re
import torchvision.transforms.functional as F

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


model_path = "/mnt/bn/multimodal-datasets-hl/llhuang/models/Qwen2.5-VL-3B-Instruct"
print(f"eval {model_path}")
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/llava_2_3m_id2video.json", 'r') as f:
    id2video_map = json.load(f)

# read data from /mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/merged_0_1800.jsonl
with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/merged_0_1800.jsonl", 'r') as f:
    previous_data = [json.loads(line) for line in f]

previous_ids = []
for item in previous_data:
    previous_ids.append(item["id"])

with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/keyframes_idx_max_128_0622_all_id_0_1800.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]


video_has_keyframes = {}

for item in data:
    id = item["id"]
    if id in previous_ids:
        continue
    vid = id[:-2]
    keyframes = item["keyframes"]
    keyframes = keyframes[:5] # only 5 items
    key_indexs = []
    for kf in keyframes:
        if kf["text"] == "null": # remove the null
            continue
        key_index = kf["key_index"]
        key_indexs.append(key_index)
    key_indexs.sort()
    if vid not in video_has_keyframes:
        video_has_keyframes[vid] = key_indexs
    else:
        video_has_keyframes[vid].extend(key_indexs)

def process_frame(image_path):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 720 * 480,
                },
                {"type": "text", "text": "Briefly describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]

# FPS_MAX_FRAMES=128
def process_video(video_path, key_indice_list):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 720 * 480,
                    "fps": 2.0,
                }
            ]
        },
    ]
    _, video_inputs = process_vision_info(messages) # list of tensor
    video_tensors = video_inputs[0] # video tensor, N C H W [0, 1]
    # import pdb; pdb.set_trace()
    top_indices = torch.tensor(key_indice_list)
    top_frames = video_tensors[top_indices]  # shape: (8, C, H, W)
    caption_dict = {}
    for idx, frame in tqdm(enumerate(top_frames)):
        frame = frame / 255
        img = F.to_pil_image(frame)
        # img.save("test.jpg")
        # import pdb; pdb.set_trace()
        keyframe_caption = process_frame(img) # pass a pil type
        caption_dict[key_indice_list[idx]] = keyframe_caption
    
    return caption_dict

with open("keyframes_caption_list_3B_0622_add_0623.jsonl", "w") as f:
    # sort keyframes
    count = 0
    for k, v in tqdm(video_has_keyframes.items()):
        # if count < 14:
        #     count += 1
        #     continue
        if(len(v)) == 0: # no keyframes need
            continue
        v = list(set(v))
        v.sort()
        video_has_keyframes[k] = v
        vid = k
        video_path = id2video_map[vid]
        video_path = os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1", video_path)
        caption_dict = process_video(video_path, v)
        save_dict = {}
        save_dict["id"] = vid
        save_dict["captions"] = caption_dict
        f.write(json.dumps(save_dict, ensure_ascii=False) + '\n')
        f.flush()
