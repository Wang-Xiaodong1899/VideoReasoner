import decord
import json
from tqdm import tqdm

# read json file
with open('train_event_id_0_3k.json', 'r') as f:
    data = json.load(f)

ok_data = []

for item in tqdm(data):
    video_path = item["segment_path"]
    
    try:
        vr = decord.VideoReader(video_path)
        # TODO: support start_pts and end_pts
        total_frames, video_fps = len(vr), vr.get_avg_fps()

        ok_data.append(item)
    except Exception as e:
        vid = item["vid"]
        print(f"{vid} reading error")

# save ok_data to json
with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_3k_exist.json", "w") as f:
    json.dump(ok_data, f, indent=4, ensure_ascii=False)