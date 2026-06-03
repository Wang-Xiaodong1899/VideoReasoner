from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import re
import json
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import fire


# model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct"
# model_path = "/mnt/bn/multimodal-datasets-hl/llhuang/models/Qwen2.5-VL-32B-Instruct"
model_path = "/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-72B-Instruct"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # using float16 on V100 GPUs
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)


def process(query, video_path):
# Messages containing a local video path and a text query

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                },
                {"type": "text", "text": query},
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
        **video_kwargs
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

    print(text)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    return output_text[0]

def main():
    # df = pd.read_csv("20250523_collect_first_42_en3.csv")
    df = pd.read_csv("去除电商0502-add25_en4.csv")
    for index, row in tqdm(df.iterrows()):
        ques = row['question_en']
        query = f"""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{ques}
{row['A']}
{row['B']}
{row['C']}
{row['D']}
The best answer is:
        """
        # video_path = os.path.join("20250501-1k", f"{row['room_id']}_{row['create_time']}_{row['end_time']}.mp4")
        video_path = os.path.join("20250502-noEC", f"{row['room_id']}_{row['create_time']}_{row['end_time']}.mp4")
        model_answer = process(query, video_path)
        # print(model_answer)
        df.loc[index, 'video_path'] = video_path
        df.loc[index, 'model_answer'] = model_answer

    # save to csv
    # df.to_csv("20250523_collect_first_42_en3_qwen2.5vl-7b.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_qwen2.5vl-32b.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_qwen2.5vl-72b.csv", index=False)
    # df.to_csv("去除电商0502-add25_en4_qwen2.5vl-7b.csv")
    # df.to_csv("去除电商0502-add25_en4_qwen2.5vl-32b.csv")
    df.to_csv("去除电商0502-add25_en4_qwen2.5vl-72b.csv")

main()