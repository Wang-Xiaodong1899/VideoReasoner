from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re

# FPS_MAX_FRAMES=128比较稳定
# 768 太多好像不太行

# FPS_MAX_FRAMES=64 python infer25.py 更为correct


model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2.5-VL-7B-Instruct"

# SFT model
model_ori = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep5-videoP80F64-N2-percentage-0706/"
# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep5-videoP80F64-N2-percentage-0706/"

# RL model
# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Charades-f80-sft3k-GRPO-clip-iou-max80-percentage-0708-N2/checkpoint-100"

# mix-trained model
model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep2-videoP80F64-percentage-no-event-N1-0709"

# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep2-videoP80F64-percentage-no-event-continue-event-ep2-N1-0709/checkpoint-100"
model_ori = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-Finetune-Zoomin-SFT/Qwen2.5-VL-7B-Instruct-Charades_v1-SFT-Video-Zoomin-3k-ep2-videoP80F64-percentage-no-event-continue-event-ep2-N1-0709-zero3-offload/"

# event


# best T-GRPO
# model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-GRPO/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-zero3-offload-0711-N3-TGRPO-Edit-fix-time/checkpoint-400"

#
model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-Prompt-Reason/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-data-TGRPO-Edit-fix-time-step250-Time-Prompt-Reasoning-GQA_Charade_ActivityNet-7k-0713-N2/checkpoint-200"

# add length prompt
model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen25-SFT-Time-Prompt-Reason/Qwen2.5-VL-7B-Instruct-Mix-data-no-pad-fix-data-TGRPO-Edit-fix-time-best400-Time-Prompt-Reasoning-GQA_Charade_ActivityNet-7k-clipmaxf16-len-prompt-0714-N2/checkpoint-200"


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
                "video": path,
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

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=False # NOTE: False to drop "<|im_start|>assistant"
# )
# edit text
# text = text + "<|im_start|>assistant><think> I want to locate the key event in the video."
# print(text)
# import pdb; pdb.set_trace()

# edit text
# text = text + "<|im_start|>assistant><think> I want to use the keyframe selection tool"
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

for _ in range(3):
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=1.0, use_cache=True)

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

    messages[1]["content"][0]["fps"] = 4 # NOTE enable fps in zoom_in video process
    # print(messages)
    print(f"truncated_text: {truncated_text}")
    import pdb; pdb.set_trace()
    # _, segment_inputs = process_vision_given_multi_durations(messages, key_durations=timeline, use_ratio=True)
    _, segment_inputs = process_vision_given_multi_durations(messages, key_durations=timeline, return_video_kwargs=False, pad_frame=False, max_frames=32, use_ratio=True)
    inputs = processor(
        text=truncated_text,
        videos=video_inputs,
        segments=segment_inputs,
        fps=messages[1]["content"][0]["fps"], # NOTE ZOOM_FPS change here
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=1.0, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("-----------------------------------")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])