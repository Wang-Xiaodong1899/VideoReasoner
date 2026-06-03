from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import re

# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2-VL-7B-Instruct"


model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2-VL-Finetune-Zoomin-SFT/Qwen2-VL-7B-zero3-offload-mix_sft_data_new_0720-N2/"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2VL-7B-Instruct-f80-GRPO-clip-iou-charades-18k-max80-ratio-0721-N2/checkpoint-800"

# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2VL-7B-Instruct-f80-GRPO-clip-iou-charades-18k-max80-ratio-0721-N2/checkpoint-1100"
model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2-SFT-Mix-GRPO-IoU-Acc/Qwen2VL-7B-Instruct-f80-Mix-GRPO-IoU-Acc-N2/checkpoint-100"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2-VL-Finetune-Zoomin-SFT/Qwen2-VL-7B-zero3-offload-mix_sft_data_new_0911-N2-trail1/checkpoint-200/"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2-SFT-Mix-GRPO-IoU-Acc/Qwen2VL-7B-Instruct-f80-Mix-SFT-0911-Acc-N2-VideoR1-0914-trail1/checkpoint-250"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2-SFT-Mix-GRPO-IoU-Acc/Qwen2VL-7B-Instruct-f80-Mix-SFT-0911-N2-GQA-iou-acc-R8-0914/checkpoint-50/"

model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct"

print(f"eval {model_path}")
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
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

new_specials = ["<|segment_pad|>", "<|event_start|>", "<|event_end|>", "<|video_zoomin|>",
                "<|keyframe_selection_tool|>", "<|keyframe_start|>", "<|keyframe_end|>",
                "<|keyframes_embed|>", "<|keyframes_pad|>"
                ]

all_specials = list(existing_specials) + new_specials

print(f"latest special tokens: {all_specials}")

print(f"old vocab size: {len(tokenizer)}") 

num_new_tokens = tokenizer.add_special_tokens({
    "additional_special_tokens": all_specials
})

print(f"new tokens: {num_new_tokens}")

print(f"New vocab size: {len(tokenizer)}")

seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>

print(f"seg_token_id: {seg_token_id}")

event_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|event_start|>") # <|segment_pad|>

print(f"event_start_token_id: {event_start_token_id}")

kf_tool_token_id = processor.tokenizer.convert_tokens_to_ids("<|keyframe_selection_tool|>") # <|segment_pad|>

print(f"kf_tool_token_id: {kf_tool_token_id}")



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
path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/lokFoo_QD8c.mp4"

# query = """
# What kind of hat does the little boy wear?
# A. Cowboy hat B. Beret C. Baseball cap D. Top hat
# """
# path = "/mnt/bn/wxd-video-understanding/wangxd/dataset/LVBench/all_videos/-hgaSElC3wU.mp4"

# query = """
# How many red socks are above the fireplace at the end of this video?
# A. 1 B. 4 C. 2 D. 3
# """
# path =  "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/fFjv93ACGo8.mp4"

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


VIDEO_QUESTION_TEMPLATE = (
        "Given a video, please analyze the content carefully and provide your response in one of the following formats:\n"
        "1. **Event localization**: locate the event using the format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where the ratios are floats between 0 and 1 indicating the relative position in the video.\n"
        "2. **Key elements extraction**: list important elements or actions in the video, output them as a comma-separated list:\n"
        "3. **Analysis and answer**: first provide an analysis, then present the final answer within the tags: <answer> </answer>.\n"
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
                "video": path,
                "max_pixels": 360 * 420,
                "fps": 2.0,
            },
            # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=query)},
            {"type": "text", "text": query}
        ]
    },
]

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True # NOTE: False to drop "<|im_start|>assistant"
# )

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False # NOTE: False to drop "<|im_start|>assistant"
)
# edit text
# text = text + "<|im_start|>assistant>\nI want to locate the key event in the video."
# print(text)

# I want to output the key elements:
# text = text + "<|im_start|>assistant>\nI want to output the key elements:"
# print(text)
# import pdb; pdb.set_trace()
# The answer is:
text = text + "<|im_start|>assistant>\nThe answer is:"
print(text)

# text = text + "<|im_start|>assistant>\n Analysis and answer:"
# print(text)

# text = text + "<|im_start|>assistant>\nI want to locate the key event in the video and output the key elements."
# print(text)

image_inputs, video_inputs, _, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
print(video_inputs[0].shape)
inputs = processor(
    text=[text],
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs # fps in video_kwargs
)
inputs = inputs.to("cuda")

for _ in range(2):
    # Inference
    import time
    start_time = time.time()

    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=1.0, use_cache=True)

    print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')
    end_time = time.time()
    print(f"inference time: {end_time - start_time}")

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
    print(truncated_text[0])

    match = re.search(r'<\|event_start\|>\[(.*?)\]<\|event_end\|>', truncated_text[0])
    if match:
        list_str = match.group(1)  # 获取 "10.6, 17.4"
        timeline = [float(num) for num in list_str.split(', ')]
        print(timeline)  # 输出: [10.6, 17.4]
        # append <|segment_pad|> with <|video_zoomin|>
        matches = list(re.finditer(r"<\|event_end\|>", truncated_text[0]))
        if len(matches) >= 2:
            print("第二个 <event_end> 的位置:", matches[1].start())
            second_index = matches[1].start()
        else:
            print("不足 2 个 <event_end>")
        truncated_text[0] = truncated_text[0][:second_index] + "<|event_end|>. Focusing on this segment <|video_zoomin|><|segment_pad|>,"
    else:
        print("未找到符合标签要求的内容")

    # messages[1]["content"][0]["fps"] = 4 # NOTE enable fps in zoom_in video process
    # # print(messages)
    # print(f"truncated_text: {truncated_text}")
    # import pdb; pdb.set_trace()
    # # _, segment_inputs = process_vision_given_multi_durations(messages, key_durations=timeline, use_ratio=True)
    # _, segment_inputs, _ = process_vision_given_multi_durations(messages, key_durations=timeline, return_video_kwargs=False, pad_frame=False, max_frames=32, use_ratio=True)
    # inputs = processor(
    #     text=truncated_text,
    #     videos=video_inputs,
    #     segments=segment_inputs,
    #     fps=messages[1]["content"][0]["fps"], # NOTE ZOOM_FPS change here
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")
    # generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=1.0, use_cache=True)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # print("-----------------------------------")
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text[0])