import os
import json
import re

def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

wrong_data = []
json_path = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/longvideoreason-train-answer/mix_sft_data_new_0911-N2-trail1-step200-f64-sampling_merged.jsonl"
with open(json_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        if item['QA']['gt_ans'] != extract_answer(item['ans']):
            wrong_data.append(item)
print(len(wrong_data))

# save to json
with open('wrong_data.json', 'w') as f:
    json.dump(wrong_data, f, indent=4)
