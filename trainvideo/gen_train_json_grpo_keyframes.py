import os
import json
from tqdm import tqdm

# origin json file
origin_json_file = "/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video_problem_id.json"

video_folder = "/mnt/bn/wxd-video-understanding/wangxd/data/Video-R1-data/"

# write json file
data = []

with open(origin_json_file, 'r') as f:
    origin_data = json.load(f)

keyframe_anno_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_0_3200.json"

with open(keyframe_anno_file, 'r') as f:
    keyframe_anno = json.load(f)

problem_id = 0
for item in tqdm(keyframe_anno):
    try:
        problem_id = item['problem_id']
        keyframes = item['keyframes']

        key_problem_id = str(problem_id)
        new_item = origin_data[key_problem_id]

        if len(keyframes) == 0:
            prompt = "<|im_start|>assistant\n<think> Let me think step by step. I don't need to detect the keyframes. "
            key_index_list = []
        else:
            key_text_index_list = keyframes
            key_index_list = []
            for key_text_index in key_text_index_list:
                key_index_list.append(key_text_index['key_index'])
            key_index_list = list(set(key_index_list))
            keyframe_len = len(key_index_list)
            key_index_list.sort()
            keyframe_pad_str = "<|keyframes_pad|>" * (keyframe_len) + ", I analyze the details provided for each frame."
            prompt = f"<|im_start|>assistant\n<think>To answer this question, I need to detect the keyframes in the video first, call the keyframe selection tool <|keyframe_selection_tool|>, and the selection result is <|keyframe_start|>{key_index_list}<|keyframe_end|>. By looking at the visual content of these keyframes <|keyframes_embed|>" + keyframe_pad_str

        keyframe_indexs = key_index_list

        new_item['prefix'] = prompt
        new_item['keyframe_indexs'] = keyframe_indexs
        new_item['path'] = os.path.join("/mnt/bn/wxd-video-understanding/wangxd/data/Video-R1-data", new_item['path'][2:])

        data.append(new_item)
        
    except Exception as e:
        print(f"{e}")
        continue

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/" + "Video-R1-kyframes-GRPO-max64.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)