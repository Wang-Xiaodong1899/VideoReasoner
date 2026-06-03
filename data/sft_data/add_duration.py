import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/train.json"

with open(json_path, 'r') as f:
    meta_data = json.load(f)

train_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_event_id_0_1k_query_time_latest_add_900_single_event_update_0708.json"

with open(train_json_path, 'r') as f:
    train_data = json.load(f)

save_data = []
for item in train_data:
    if "vid" in item:
        vid = item['vid']
        metadata = meta_data[vid]
        duration = metadata['duration']
        times = item['times']
        start_time = times[0]
        end_time = times[1]
        start_point = start_time / duration
        end_point = end_time / duration

        # 保留2位小数
        start_point = round(start_point, 2)
        end_point = round(end_point, 2)

        item["start_point"] = start_point
        item["end_point"] = end_point
        item["points"] = [start_point, end_point]
        item["duration"] = duration
    save_data.append(item)

with open(train_json_path, 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)