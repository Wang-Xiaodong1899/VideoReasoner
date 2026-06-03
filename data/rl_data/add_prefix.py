import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_find_query_train_grpo_iou_only.json"

with open(json_path, "r") as f:
    data = json.load(f)

for item in data:
    item["prefix"] = "I want to locate the key event in the video. "

new_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_find_query_train_grpo_iou_only_prefix.json"

with open(new_json_path, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
