from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
import re
import json
import os
from tqdm import tqdm
import csv
import fire

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/activitynet/train_grpo_solution_Charades_v1_activitynet.json", "r") as f:
    train_data = json.load(f)


model_path = "/mnt/bn/wxd-video-understanding/wangxd/ckpt/Qwen2.5-VL-7B-Instruct-Charades-Activity-f64-sft3k-GRPO-clip-iou-max64-0701"

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

existing_specials = tokenizer.special_tokens_map.get("additional_special_tokens", [])
print(f"current special tokens: {existing_specials}")


print(f"old vocab size: {len(tokenizer)}") 


seg_token_id = processor.tokenizer.convert_tokens_to_ids("<|segment_pad|>") # <|segment_pad|>

print(f"seg_token_id: {seg_token_id}")

VIDEO_QUESTION_TEMPLATE = (
    "Give you a video. Please think about your reasoning before answering the question.\n"
    "If the content of the question in the video is unclear or difficult to see, first locate the key event by generating the start and end times in the following format: <|event_start|> [t_start, t_end] <|event_end|>.\n"
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

    # print(truncated_text[0])

    # match = re.search(r'<\|event_start\|>\[(.*?)\]<\|event_end\|>', truncated_text[0])
    # 找出所有 <think> 和 <\|event_start\|> 的匹配位置
    text = truncated_text[0]
    think_matches = list(re.finditer(r"<think>", text))
    event_start_matches = list(re.finditer(r"<\|event_start\|>", text))

    # 确保至少有两个 <think> 和两个 <|event_start|>
    if len(think_matches) >= 2 and len(event_start_matches) >= 2:
        start = think_matches[1].start()
        end = event_start_matches[1].end()  # 包括标签本身
        # print(text[start:end])
        return text[start:end]
    else:
        return ""


def main(start, end):
    test_data = train_data[start:end]

    our_pred = []
    with open(f"stage_1_IoU_RFModel_pred_{start}_{end}.jsonl", "w") as f:
        for item in tqdm(test_data):
            video_path = item['path']
            query = item['problem']
            output = process(query, video_path)
            item["iou_reason"] = output
            our_pred.append(item)
            # save to jsonl file
            json.dump(our_pred[-1], f)
            f.write('\n')
            f.flush()
    # save our_pred to json
    with open(f"stage_1_IoU_RFModel_pred_{start}_{end}.json", "w") as f:
        json.dump(our_pred, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    fire.Fire(main)