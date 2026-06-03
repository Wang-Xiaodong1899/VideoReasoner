import json
import math
import pandas as pd

json_path = "mix_sft_data_new_0720.json"

with open(json_path, "r") as f:
    data = json.load(f)

charader_data = []
llava_data = []
for item in data:
    if item["data_source"] == "Charades_v1":
        charader_data.append(item)
    elif item["data_source"] == "llava_178k_2_3m":
        llava_data.append(item)

df = pd.read_csv("keyframe_Charades_sft_data_seed16vl.csv")
keyitems_str = df["seed16vl"].tolist()

save_data = []
for idx, item in enumerate(charader_data):
    if isinstance(keyitems_str[idx], float):
        continue
    item["keyitems"] = [item.strip().strip() for item in keyitems_str[idx].split(',')]
    save_data.append(item)

df = pd.read_csv("keyframe_sft_data_seedvl16.csv")
keyitems_str = df["seed16vl"].tolist()

for idx, item in enumerate(llava_data):
    if isinstance(keyitems_str[idx], float):
        continue
    item["keyitems"] = [item.strip().strip() for item in keyitems_str[idx].split(',')]
    save_data.append(item)

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/RL_GQA_single_segment_data.json"

with open(json_path, "r") as f:
    GQA_data = json.load(f)

df = pd.read_csv("keyframe_GQA_single_sft_data_seedvl16.csv")
keyitems_str = df["seed16vl"].tolist()

for idx, item in enumerate(GQA_data):
    if isinstance(keyitems_str[idx], float):
        continue
    item["keyitems"] = [item.strip().strip() for item in keyitems_str[idx].split(',')]
    save_data.append(item)

new_data = []
for item in save_data:
    if len(item["keyitems"]) == 0:
        continue
    new_data.append(item)

# save_data to json file
with open("mix_keyframes_Charades_llava_GQA_data_new_0721.json", "w") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
