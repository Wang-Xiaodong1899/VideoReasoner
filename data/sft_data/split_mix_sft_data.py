import json


data_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_event_id_0_1k_query_time_latest_add_900_single_event_update_0708.json"

with open(data_path, "r") as f:
    data = json.load(f)

save_data = []

for item in data:
    if item["data_source"] != "Charades_v1":
        save_data.append(item)

# save data
# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_event_id_0_1k_query_time_latest_update_0708.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/train_900_single_event_update_0708.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)

