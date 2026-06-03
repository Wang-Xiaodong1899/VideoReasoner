import json
from tqdm import tqdm

from qwen_vl_utils import fetch_video, _read_video_decord


# path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_1_qa_01_grpo_solution.json"

# max_frames = 80
# output_path = f"/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_1_qa_01_grpo_solution_time2frame_{max_frames}.json"

path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/activitynet/train_grpo_solution_Charades_v1_activitynet_0704.json"
max_frames = 80
output_path = f"/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/activitynet/train_grpo_solution_Charades_v1_activitynet_0704_time2frame_{max_frames}.json"

with open(path, 'r') as f:
    data = json.load(f)

save_data = []
for item in tqdm(data):
    times = item["times"] # float list
    path = item["path"]
    ele = {
        'video': path,
        'max_frames': max_frames,
        'return_nframes': True
    }
    _, sample_fps, nframes = _read_video_decord(ele)
    item["sample_fps"] = sample_fps
    item["nframes"] = nframes
    duration = item["duration"]
    # times to frames
    # times[0], times[1] in [0, duration] with nframes
    # start_frame, end_frame
    start_frame = int(times[0] * sample_fps)
    end_frame = int(times[1] * sample_fps)
    if start_frame < 0:
        start_frame = 0
    if end_frame > nframes:
        end_frame = nframes
    item["frames"] = [start_frame, end_frame]

    save_data.append(item)

with open(output_path, 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
