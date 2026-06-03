# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info, process_vision_segment_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""
    VIDEO_QUESTION_TEMPLATE = (
        "Give you a video. Please think about your reasoning before answering the question.\n"
        "If the content of the question in the video is unclear or difficult to see, first locate the key event by generating its location as a proportion of the video duration in the following format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where start_ratio and end_ratio are floats between 0 and 1, indicating the relative start and end points of the key event.\n"
        "Then, use the <|video_zoomin|> tag to zoom in on the event content for closer inspection.\n"
        "Please show your reasoning and answer in the following tags: <think> reasoning process here </think> <answer> answer here </answer>\n"
        "Question: {Question}\n"
    )
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']
    
    QUESTION_TEMPLATE = VIDEO_QUESTION_TEMPLATE

    system_message = "You are a very smart multimodal assistant that can understand videos. "

    # 构建基础 user content
    user_content = [
        {
            "type": "video",
            "video": example['path'],
            "max_pixels": 80*28*28,
            "max_frames": 64
        },
        {
            "type": "text",
            "text": QUESTION_TEMPLATE.format(Question=question)
        }
    ]

    # 如果有 segment_path 则添加
    if 'segment_path' in example and example['segment_path']:
        user_content.append({
            "type": "segment",
            "video": example['segment_path'],
            "max_pixels": 100*28*28,
            "max_frames": 16,
        })

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "\n" + example['response']}]
        }
    ]
    
    return {"messages": messages, "duration": example['duration'], "times": example['times'], "points": example['points']}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            image_inputs, video_inputs, segment_inputs, video_kwargs = process_vision_segment_info(example["messages"], return_video_kwargs=True)
            
            # use nframes and fps to frames

            # nframes = video_inputs[0].shape[0]
            # duration = example['duration']
            # fps = nframes / duration

            points = example['points']

            # edit example["messages"]
            res_content = example["messages"][2]["content"]
            prefix = res_content[0]["text"].split("<|event_start|>")[0]
            tail = res_content[0]["text"].split("<|event_end|>")[1]
            example["messages"][2]["content"][0]["text"] = prefix + f"a proportion of <|event_start|>[{points[0]:.2f}, {points[1]:.2f}]<|event_end|>" + tail
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            print(f"{texts=}")

        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        segments=segment_inputs,
        return_tensors="pt",
        padding=True,
        **video_kwargs
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # <|im_start|>: 151644
    for i in range(labels.size(0)):
        label_seq = labels[i]
        last_pos = (label_seq == 151644).nonzero()[-1]
        label_seq[:last_pos+3] = -100 # mask all tokens before <|im_start|>assistant\n\n
        labels[i] = label_seq

    # Handle visual tokens based on processor type
    # NOTE visual tokens missing 151655
    # visual_tokens = [151652, 151653, 151655, 151656] if isinstance(processor, Qwen2VLProcessor) else [
    #     processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    # ]
    # visual_tokens = [151652, 151653, 151655, 151656] # Qwen-VL tokens
    
    visual_tokens = [processor.tokenizer.convert_tokens_to_ids(token) for token in ["<|segment_pad|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]]

    print(f"visual token: {visual_tokens}")

    # import pdb; pdb.set_trace()

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100
    


    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(), # comment it if use zero-3
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    # if Instruct not in model_config.model_name_or_path:
    if "Qwen2-VL" in model_config.model_name_or_path and "Instruct" not in model_config.model_name_or_path:
        processor = AutoProcessor.from_pretrained(
            "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen2-VL-7B-Instruct",
            trust_remote_code=model_config.trust_remote_code
        )
        print("using Qwen2-VL-Instruct processor")
    else:
        processor = AutoProcessor.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code
        )
    
    # import pdb; pdb.set_trace()

    tokenizer = processor.tokenizer

    existing_specials = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    print(f"current special tokens: {existing_specials}")

    new_specials = ["<|segment_pad|>", "<|event_start|>", "<|event_end|>", "<|video_zoomin|>"]

    all_specials = list(existing_specials) + new_specials

    print(f"latest special tokens: {all_specials}")

    print(f"old vocab size: {len(tokenizer)}") 

    num_new_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": all_specials
    })

    print(f"new tokens: {num_new_tokens}")

    print(f"New vocab size: {len(tokenizer)}")

    model.config.seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>
    
    print(f"model.config.seg_token_id: {model.config.seg_token_id}")

    x_token_id = processor.tokenizer.convert_tokens_to_ids("<|event_start|>") # <|segment_pad|>
    print(f"<|event_start|>: {x_token_id}")
    x_token_id = processor.tokenizer.convert_tokens_to_ids("<|event_end|>") # <|segment_pad|>
    print(f"<|event_end|>: {x_token_id}")
    x_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_zoomin|>") # <|segment_pad|>
    print(f"<|video_zoomin|>: {x_token_id}")

    # import pdb; pdb.set_trace()
    
    # NOTE change chat_template
    # if hasattr(processor, "tokenizer"):
    # Qwen2VL
    processor.chat_template = """{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
{% for message in messages %}
  {% if loop.first and message['role'] != 'system' %}
<|im_start|>system
You are a helpful assistant.<|im_end|>
  {% endif %}
<|im_start|>{{ message['role'] }}
  {% if message['content'] is string %}
{{ message['content'] }}<|im_end|>
  {% else %}
    {% for content in message['content'] %}
      {# 忽略 segment 类型的 content #}
      {% if content['type'] == 'segment' %}
        {# skip segment #}
      {% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}
        {% set image_count.value = image_count.value + 1 %}
        {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
<|vision_start|><|image_pad|><|vision_end|>
      {% elif content['type'] == 'video' or 'video' in content %}
        {% set video_count.value = video_count.value + 1 %}
        {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
<|vision_start|><|video_pad|><|vision_end|>
      {% elif 'text' in content %}
{{ content['text'] }}
      {% endif %}
    {% endfor %}
<|im_end|>
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}

"""
    # import pdb; pdb.set_trace()

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer # comment it if using trl==0.16.0, else trl==0.14.0
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    # NOTE add
    processor.tokenizer.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
