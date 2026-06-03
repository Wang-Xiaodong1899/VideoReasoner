import json
import os
import pandas as pd

video_dir = "/mnt/bn/wxd-video-understanding/wangxd/data/LongVideo-Reason/longvila_videos"

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/longvideoreason/train_exist_25744.json"
with open(json_path, 'r') as f:
    longvideoreason_data = json.load(f)

problem_id = 0
save_data = []
# 0911: 0-12000
for item in longvideoreason_data[:12000]:
    # key items
    # accuracy
    problem = item["problem"]
    data_type = item["data_type"]
    videos = item["videos"]
    path = os.path.join(video_dir, videos)
    answer = item["answer"]
    # answer = item["answer"].replace("<answer>", "").replace("</answer>", "").strip()
    answer_symbol = item["answer"].replace("<answer>", "").replace("</answer>", "").strip()
    options = problem.split("\n")
    options = [option.strip() for option in options if option.strip() != ""][1:]
    option_dict = {}
    for option in options:
        option_dict[option[0]] = option
    try:
        complete_answer = option_dict[answer_symbol]
    except:
        print(answer_symbol, option_dict)
        # import pdb; pdb.set_trace()
        continue
    times = []
    duration = 0
    data_source = 'longvideoreason'
    save_data.append({
        "problem_id": problem_id,
        "times": times,
        "problem": problem,
        "data_type": data_type,
        "problem_type": "qa",
        "prefix": "Analysis and answer: ",
        "options": options,
        "solution": answer,
        "path": path,
        "data_source": data_source,
        "duration": duration,
    })
    problem_id += 1
    # import pdb; pdb.set_trace()


with open(f"data/rl_data/longvideoreason_qa_{len(save_data)}_0911.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)