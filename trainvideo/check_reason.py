import json
import re
from tqdm import tqdm
from doubao import chat_text

def extract_think(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_3k_exist.json", "r") as f:
    data = json.load(f)

with open("failed_reasoning.jsonl", "w") as f:
    for item in tqdm(data):
        problem_id = item["problem_id"]
        problem = item["problem"]
        response = item["response"]
        think = extract_think(response)

        prompt = f"""
        You are given a question and an answer. 
        Please judge whether the answer at the beginning of the answer is strongly related to the question.
        If so, output Yes, otherwise output No.
        Question: {problem}
        answer: {think}
        """
        
        result_str = chat_text(prompt, "doubao-1-5-thinking-pro-250415")
        result_str = result_str.strip()

        item["judge"] = result_str

        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
