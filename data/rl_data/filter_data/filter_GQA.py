import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/GQA_longvideoreason_ground_qa_36178.json"
with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
for item in data:
    if item["data_source"] == "GQA":
        solution = item["solution"]
        # add <answer> </answer>
        solution = "<answer>" + solution.strip()[0] + "</answer>"
        item["solution"] = solution
        save_data.append(item)

new_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/filter_data/filter_GQA_qa_ground.json"
with open(new_json_path, "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)