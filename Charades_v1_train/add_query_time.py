import json
from tqdm import tqdm

all_data_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_3k_query_time.json"
pick_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_no_1k3.json"

all_dict_vid_query_times = {}

with open(all_data_path, 'r') as f:
    all_data = json.load(f)

for item in all_data:
    vid = item['vid']
    all_dict_vid_query_times[vid] = [
        item['query'],
        item['times']
    ]

with open(pick_path, 'r') as f:
    pick_data = json.load(f)

for idx, item in tqdm(enumerate(pick_data)):
    vid = item['vid']
    new_ = all_dict_vid_query_times[vid]
    pick_data[idx]['query'] = new_[0]
    pick_data[idx]['times'] = new_[1]

# save pick_data
save_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_no_1k3_times.json"

with open(save_path, 'w') as f:
    json.dump(pick_data, f, indent=4, ensure_ascii=False)


