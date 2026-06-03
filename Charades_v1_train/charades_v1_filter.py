import json

data_path = "charades_v1_train_grpo_iou_only.json"

with open(data_path, 'r') as f:
    data = json.load(f)

for idx, item in enumerate(data):
    times = item['times']
    duration = item['duration']
    if times[-1] > duration:
        times[-1] = duration
    if times[0] < 0:
        times[0] = 0
    data[idx] = item

with open(data_path, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
