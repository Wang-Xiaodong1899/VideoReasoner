import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import re

# Set model path
model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Video-R1-7B-Qwen2.5-VL/"

# Set video path and question
video_path = "./src/example_video/video1.mp4"
question = "Which move motion in the video lose the system energy?"

question = """
What is the number of the first lipstick she used?"
A. 600, B. 656, C. 866, D. 999
"""
video_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/lokFoo_QD8c.mp4"


# Choose the question type from 'multiple choice', 'numerical', 'OCR', 'free-form', 'regression'
problem_type = 'free-form'

# Initialize the LLM
if "Qwen2-VL" in model_path or "qwen2_vl" in model_path or "Qwen2VL" in model_path:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
        attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
        device_map="auto",
    )
elif "Qwen2.5-VL" in model_path:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
        attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
        device_map="auto",
    )

processor = AutoProcessor.from_pretrained(model_path)

tokenizer = processor.tokenizer

# Load processor and tokenizer
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

# Prompt template
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# Question type 
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

# Construct multimodal message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 200704, # max pixels for each frame
                "nframes": 64 # max frame number
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[problem_type]
            },
        ],
    }
]

# Convert to prompt string
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process video input
image_inputs, video_inputs, _, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

inputs = processor(
    text=[prompt],
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs # fps in video_kwargs
)
inputs = inputs.to("cuda")

import time
start_time = time.time()

generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)

print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(generated_ids_trimmed[0].shape)
outputs_1 = output_text[0]
print(f"response: {outputs_1}")

print(f"inference time: {time.time() - start_time}")
