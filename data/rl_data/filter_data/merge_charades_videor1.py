import json
import os


json_path1 = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_longvideoreason_ground_qa_49333.json"

with open(json_path1, "r") as f:
    data1 = json.load(f)

save_data = []
for item in data1:
    if item['data_source'] == 'Charades_v1':
        save_data.append(item)

# json_path2 = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/Video-R1-260k-filter-video-26533-0913.json"

# with open(json_path2, "r") as f:
#     data2 = json.load(f)

# for item in data2:
#     save_data.append(item)

# new_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/Charades_v1-18k-Video-R1-260k-filter-video-26533-0913.json"
# # save data
# with open(new_json_path, "w") as f:
#     json.dump(save_data, f, ensure_ascii=False, indent=4)

new_json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/Charades_v1-18k-prefix.json"
with open(new_json_path, "w") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=4)
