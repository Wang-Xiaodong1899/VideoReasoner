import pandas as pd
import json
import re
from tqdm import tqdm

# df to json
df = pd.read_csv("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/keyframes_caption_list_3B_0623_prompt_output_deepseek.csv")
json_data = df.to_json(orient="records")

# each item reerite to 
# "problem_id": 0,
# "vid": "00HFP",
# "query": "person throwing them on the floor.",
# "times": [
#     13.6,
#     21.9
# ],
# "problem": "In the segment where the person throws the shoes on the floor, in what manner are the shoes tossed?",
# "data_type": "video",
# "problem_type": "multiple choice",
# "options": [
#     "A. Only to the left",
#     "B. Only upwards",
#     "C. In various directions",
#     "D. Straight ahead"
# ],
# "answer": "C",
# "path": "/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1/00HFP.mp4",
# "segment_path": "/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1_segments_2/00HFP_1.mp4",
# "response":

def extract(text):
    # 提取问题
    question_match = re.search(r'### Question and Options\s*\n\n(.*?)\nA\.', text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # 提取选项
    options_match = re.findall(r'([A-D])\. (.*?)\n', text)
    options = [f"{label}. {opt}" for label, opt in options_match[:4]]

    # 提取关键帧列表
    keyframes_match = re.search(r'The key frames detected are \[([0-9,\s]+)\]', text)
    keyframes = list(map(int, keyframes_match.group(1).split(','))) if keyframes_match else []

    # 提取关键帧描述 JSON（注意花括号嵌套，需要处理两层）
    # frame_desc_match = re.search(r'detailed description of each key frame:\n\{{(.*?)\}}\}', text, re.DOTALL)
    # frame_desc_json = "{" + frame_desc_match.group(1) + "}}" if frame_desc_match else "{}"
    # frame_descriptions = json.loads(frame_desc_json)

    # 提取答案
    answer_match = re.search(r'### Answer\s*([A-D])', text)
    answer_letter = answer_match.group(1) if answer_match else ""
    answer = ""

    # 根据字母去 options 中匹配完整选项
    if answer_letter:
        for opt in options:
            if opt.startswith(f"{answer_letter}."):
                answer = opt
                break
    
    return question, options, keyframes, answer

json_list = json.loads(json_data)

problem_id = 0
save_data = []
for item in tqdm(json_list):
    content = item["question"]
    question, options, keyframes, answer = extract(content)
    # check item["deepseek"]
    response = item["deepseek"]
    if response.count("<|keyframe_selection_tool|>") !=1:
        print(f"<|keyframe_selection_tool|> count not 1")
        continue
    if response.count("<|keyframe_start|>") !=1:
        print(f"<|keyframe_start|> count not 1")
        continue
    if response.count("<|keyframe_end|>") !=1:
        print(f"<|keyframe_end|> count not 1")
        continue
    if response.count("<|keyframes_embed|>") !=1:
        print(f"<|keyframes_embed|> count not 1")
        continue

    keyframe_length = len(keyframes)
    response = response.replace("<|keyframes_embed|>", "<|keyframes_embed|>"+"<|keyframes_pad|>"*keyframe_length)

    response = "<think>" + response + "</think>"
    response = response + "<answer>" + answer + "</answer>"
    print(answer)

    # import pdb; pdb.set_trace()
    save_item = {
        "problem_id": problem_id,
        "vid": item["id"],
        "path": item["video_path"],
        "problem": question,
        "data_type": "video",
        "problem_type": "multiple choice",
        "options": options,
        "answer": answer,
        "response": response,
        "data_source": "llava_178k_2_3m",
        "keyframes": keyframes
    }
    problem_id += 1
    save_data.append(save_item)

# save to json
with open("trainvideo/llm_train_tool_use.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
