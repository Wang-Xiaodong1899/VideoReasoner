import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_1k_query_time_latest_add_900_single_event.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
for item in data:
    response = item["response"]
    if "<|event_start|>" in response:
        tail = response.split("<think>")[1].strip()
        prefix = "<think> I want to locate the key event in the video. "
        output = prefix + tail
    else:
        tail = response.split("<think>")[1].strip()
        prefix = "<think> Based on the existing video, I can proceed directly to the next reasoning. "
        output = prefix + tail
    item["response"] = output
    save_data.append(item)

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_event_id_0_1k_query_time_latest_add_900_single_event_update_0708.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
