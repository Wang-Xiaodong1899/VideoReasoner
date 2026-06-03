import os
import json
import glob
from tqdm import tqdm

json_dir = "anno/"

# search ground_2QA*.json
files = glob.glob(os.path.join(json_dir, "grounding_2QA*.json"))
train_path = "train_filter.json"

with open(train_path, "r") as f:
    train_data = json.load(f)

save_data = []
problem_id = 0
for file in tqdm(files):
    vid = file.split("/")[-1].split(".")[0].split("QA_")[-1]
    with open(file, "r") as f:
        data = json.load(f)
    train_instance = train_data[vid]
    timestamps = train_instance["timestamps"]
    duration = train_instance["duration"]
    sentences = train_instance["sentences"]
    idx = 0
    for sen, ans in data.items():
        gt_sen = sentences[idx]
        if sen != gt_sen:
            continue
        QAs = ans
        timestamp = timestamps[idx] # [0, 11.89]
        for QA in QAs: # 2 QAs
            ques = QA["question"]
            opts = QA["options"]
            ans = QA["answer"]
            save_item = {
                "problem_id": problem_id,
                "vid": vid,
                "query": sen,
                "times": timestamp,
                "problem": ques,
                "data_type": "video",
                "problem_type": "multiple choice",
                "options": opts,
                "answer": ans,
                "path": os.path.join("/mnt/bn/wxd-video-understanding/wangxd/dataset/activity-caption/ActivityNet_Captions/video/", vid+".mp4"),
                "data_source": "ActivityNet",
                "solution": "<answer>"+ans+"</answer>",
                "duration": duration,
            }
            save_data.append(save_item)
            problem_id += 1

with open("train_grpo_solution_0704.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)


