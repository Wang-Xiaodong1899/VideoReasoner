import sys
import os
import importlib.util
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import fire
import torch
import math
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")


# from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
# import re

from perception_encoder.keyframe_api import process, process_queries, get_video_embeddings, get_query_embedding, search_indices, get_queries_embedding

subset = "short"

# embeddings_dir = f"/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videomme_{subset}_f768"

embeddings_dir = f"/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videomme_{subset}_f128"

# embeddings_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/lvbench_f768"

# 获取question和video_path
# videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/benchmarks/videomme-{subset}-ques-keyitem-qwenf128.csv"

# videomme_csv = f"/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/videomme-{subset}-ques-EGRPO.csv"

# videomme_csv = f"/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/videomme_{subset}_key_doubao16-0615_sort.csv"

# lvbench_csv = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/lvbench_key_doubao16-0615_sort.csv"

# 0920
videomme_csv = f"/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/videomme_{subset}_noprefix-keyframes-0920.csv"

# df = pd.read_csv(lvbench_csv)
df = pd.read_csv(videomme_csv)

video_paths = df["video_path"].tolist()

questions = df["question"].tolist()


# keyitems = df["qwenvl25_7b"].tolist() # TODO keyitem

# keyitems = df["qwenvl25_7b_mix_temp"].tolist() # TODO keyitem

# keyitems = df["qwenvl25_7b_egrpo"].tolist()

# keyitems = df["seed16vl"].tolist()

# 0920
keyitems = df["qwenvl2_7b_mix_fix_data_temp_mix_grpo_key"].tolist()

video_array_dict = {}

for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        file_path = os.path.join(embeddings_dir, file)
        video_embeddng_array = np.load(file_path)
        video_array_dict[file.replace(".npy", "")] = video_embeddng_array

save_data = []
for idx, keyitem in tqdm(enumerate(keyitems)):
    keyitem = str(keyitem)
    input_items = keyitem.split(",")
    # 如果input_items不是list，转成list
    if not isinstance(input_items, list):
        input_items = [input_items]
    # input_items = input_items[:16] # 只取前16个
    # input_items = input_items[:8] # 只取前8个
    max_select_items = 8
    input_items = input_items[:max_select_items]
    input_items = [x.strip() for x in input_items]
    queries, text_embedding = get_queries_embedding(input_items)
    video_path = video_paths[idx]
    videoname = video_path.split("/")[-1].split(".")[0]
    video_embedding_array = video_array_dict[videoname]
    query_len = len(input_items)
    # query_k = math.ceil(64//query_len) # top-64
    query_k = 1 # hard code for 2
    segment_len = 32
    # array to tensor
    video_embedding = torch.from_numpy(video_embedding_array).cuda()
    idx_list = search_indices(video_embedding, queries, text_embedding, segment_len=segment_len, k=query_k)
    total_frames = video_embedding_array.shape[0]
    item = {
        'total_frames': total_frames,
        'idx': idx,
        'ques': questions[idx],
        'keyitem': keyitem,
        'video_path': video_path,
        'idx_list': idx_list
    }
    save_data.append(item)

# save_data to json file

# with open(f"videomme_{subset}_f768_qwenf128_keyframe_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_{subset}_f128_our-EGRPO_keyframe_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_{subset}_f768_our-EGRPO_keyframe_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_{subset}_768_our-EGRPO_keyframe_query16_topk2_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_{subset}_f768_our-EGRPO_keyframe_query8_topk4_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"lvbench_f768_our-EGRPO_keyframe_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"lvbench_f768_our-EGRPO_keyframe_query8_topk4_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"lvbench_f768_seed16vl_keyframe_query16_topk4_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_short_f128_seed16vl_keyframs_query16_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_short_f128_seed16vl_segment_query16_seg{segment_len}_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"videomme_short_f768_seed16vl_segment_query16_seg{segment_len}_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# with open(f"lvbench_f768_seed16vl_segment_query{max_select_items}_seg{segment_len}_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# 0920
# with open(f"0920_lvbench_f768_seed16vl_segment_query{max_select_items}_seg{segment_len}_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)


# medium, long
# with open(f"0920_videomme_{subset}_f768_seed16vl_segment_query{max_select_items}_seg{segment_len}_topk{query_k}_index.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

with open(f"0920_videomme_{subset}_f128_seed16vl_segment_query{max_select_items}_seg{segment_len}_topk{query_k}_index.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)