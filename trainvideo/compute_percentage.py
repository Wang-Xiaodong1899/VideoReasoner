import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_3k_query_time_0705_check.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
for item in data:
    duraiton = item["duration"]
    times = item["times"]
    start_time = times[0]
    end_time = times[1]
    start_point = start_time / duraiton
    end_point = end_time / duraiton

    # 保留2位小数
    start_point = round(start_point, 2)
    end_point = round(end_point, 2)

    item["start_point"] = start_point
    item["end_point"] = end_point

    item["points"] = [start_point, end_point]
    save_data.append(item)

with open(json_path, "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)

