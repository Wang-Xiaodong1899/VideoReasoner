import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_v1_train_grpo_iou_only.json"

with open(json_path, "r") as f:
    data = json.load(f)

problems = []
for item in data:
    problems.append(item['problem'])

print(len(problems))

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_find_query_train_grpo_iou_only_prefix.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
cnt = 0
for item in data:
    problem = item['problem']
    if problem in problems:
        item['problem_type'] = 'qa'
        cnt += 1
    else:
        item['problem_type'] = 'grounding'
    save_data.append(item)

print(cnt)

new_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_find_query_train_grpo_iou_only_prefix_qa_grounding.json"

with open(new_json_path, "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
