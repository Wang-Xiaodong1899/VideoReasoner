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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")


# from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
# import re

from perception_encoder.keyframe_api import process, process_queries, get_video_embeddings, get_query_embedding, search_indices, get_queries_embedding

embeddings_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videomme_long_f768"

# 获取question和video_path
videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/benchmarks/videomme-long-ques-event.csv"

df = pd.read_csv(videomme_csv)

video_paths = df["video_path"].tolist()

questions = df["question"].tolist()

video_array_dict = {}

for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        file_path = os.path.join(embeddings_dir, file)
        video_embeddng_array = np.load(file_path)
        video_array_dict[file.replace(".npy", "")] = video_embeddng_array

save_data = []
for idx, question in tqdm(enumerate(questions)):
    queries, text_embedding = get_query_embedding(question)
    video_path = video_paths[idx]
    videoname = video_path.split("/")[-1].split(".")[0]
    video_embedding_array = video_array_dict[videoname]
    # array to tensor
    video_embedding = torch.from_numpy(video_embedding_array).cuda()
    idx_list = search_indices(video_embedding, queries, text_embedding)
    item = {
        'ques': question,
        'idx': idx,
        'video_path': video_path,
        'idx_list': idx_list
    }
    save_data.append(item)

# save_data to json file
import json
with open("videomme_long_f768_keyframe_index.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)

