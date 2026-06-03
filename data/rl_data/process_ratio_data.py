import json

data_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/stage_1_IoU_RLModel_pred_GT_time.json"

with open(data_path, 'r') as f:
    data = json.load(f)

save_data = []
for item in data:
    times = item['times']
    duration = item['duration']
    think = item['think']
    ratios = [ti/duration for ti in times]
    ratios = [1.0 if ratio > 1.0 else ratio for ratio in ratios]
    prefix = think.split('<|event_start|>')[0]
    prefix = prefix.replace('<think>', '<think>I want to locate the key event in the video.')
    tail = think.split('<|event_end|>')[1]
    think = prefix + f"a proportion of <|event_start|>[{', '.join([f'{ratio:.2f}' for ratio in ratios])}]<|event_end|>" + tail
    item['think'] = think
    item['points'] = ratios
    save_data.append(item)

with open('/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/stage_1_IoU_RLModel_pred_GT_time_ratio.json', 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
