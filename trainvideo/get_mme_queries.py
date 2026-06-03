import sys
import os
import importlib.util
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")

from perception_encoder.keyframe_api import process, process_queries, get_video_embeddings


def main(start=0, end=300):

    # videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/benchmarks/videomme-long-ques-event.csv"
    videomme_csv = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/lvbench.csv"

    df = pd.read_csv(videomme_csv)

    video_paths = df["video_path"].tolist()

    video_paths = list(set(video_paths))

    print(len(video_paths))
    # import pdb; pdb.set_trace()

    # save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/videomme_long_f768/"
    save_dir = "/mnt/bn/wxd-video-understanding/wangxd/eval/Open-R1-Video-V1/eval/pe_embeddings/lvbench_f768/"

    os.makedirs(save_dir, exist_ok=True)

    for video_path in tqdm(video_paths[start:end]):
        savename = os.path.basename(video_path).replace(".mp4", ".npy")
        savepath = os.path.join(save_dir, savename)
        if os.path.exists(savepath):
            continue
        video_embeddings = get_video_embeddings(video_path) # torch.tensor
        video_embeddings = video_embeddings.cpu().numpy() # numpy.ndarray
        np.save(savepath, video_embeddings)

if __name__ == "__main__":
    fire.Fire(main)
