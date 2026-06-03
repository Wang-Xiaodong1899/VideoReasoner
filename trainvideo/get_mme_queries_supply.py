import sys
import os
import importlib.util
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import fire
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")


# from qwen_vl_utils import process_vision_info, process_vision_given_multi_durations
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
# import re

from perception_encoder.keyframe_api import process, process_queries, get_video_embeddings


def main(start=0, end=None):

    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/benchmarks/videomme-long-ques-event.csv"
    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/lvbench.csv"
    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/longvideobench.csv"
    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/mlvu.csv"
    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/videoevalpro_mcq.csv"
    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/vsibench.csv"
    videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/repo/benchmark-video-llms-eval/mmvu_noprefix.csv"

    df = pd.read_csv(videomme_csv)

    video_paths = df["video_path"].tolist()

    video_paths = list(set(video_paths)) # 每次set都是无序的
    video_paths.sort()

    print(f"process {len(video_paths)} videos")

    # import pdb; pdb.set_trace()

    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videomme_long_f768/"
    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/lvbench_f768/"
    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/longvideobench_f768/"
    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/mlvu_f768/"
    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videoevalpro_mcq_f768/"
    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/vsibench_f768/"
    save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/mmvu_f768/"


    os.makedirs(save_dir, exist_ok=True)
    todo = []
    for video_path in tqdm(video_paths[start:end]):
        # savename = os.path.basename(video_path).replace(".mp4", ".npy")
        # for MMVU
        savename = video_path.split('MMVU/videos/')[-1].replace('/', '_').replace('.mp4', '.npy')
        savepath = os.path.join(save_dir, savename)
        if os.path.exists(savepath):
            continue
        video_embeddings = get_video_embeddings(video_path) # torch.tensor
        video_embeddings = video_embeddings.cpu().numpy() # numpy.ndarray
        np.save(savepath, video_embeddings)
        # todo.append(video_path)
        # import pdb; pdb.set_trace()
    # print(len(todo))
    # import pdb; pdb.set_trace()
    # # # save todo to json file
    # todo.sort()
    
    # with open(os.path.join("todo_longvb.json"), "w") as f:
    #     json.dump(todo, f, indent=4)
    # import pdb; pdb.set_trace()
    # with open(os.path.join("todo_mlvu.json"), "r") as f:
    #     todo = json.load(f)
    

    # for video_path in tqdm(todo[start:end]):
    #     savename = os.path.basename(video_path).replace(".mp4", ".npy")
    #     savepath = os.path.join(save_dir, savename)
    #     if os.path.exists(savepath):
    #         continue
    #     video_embeddings = get_video_embeddings(video_path) # torch.tensor
    #     video_embeddings = video_embeddings.cpu().numpy() # numpy.ndarray
    #     np.save(savepath, video_embeddings)


if __name__ == "__main__":
    fire.Fire(main)
