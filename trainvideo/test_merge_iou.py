import re
import pandas as pd
import ast
import json

def extract_time_interval(text):
    # 匹配类似 "0.60 - 5.10" 或 "18.00 to 32.00"
    pattern = r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, text)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return [start, end] if start < end else [end, start]
    return None

def compute_iou(pred, gt):
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0

def main():
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/our_pred_0.jsonl"
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/our_pred_reverse.jsonl"
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_sft_pred_all.jsonl"
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_rl_200_pred_100.jsonl"
    json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_mix_rl_150_VMax768_pred_100.jsonl"
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_mix_rl_150_VMax768_pred_100_fix_max.jsonl"
    json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_rl_600_VMax768_pred_100.jsonl"
    json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_rl_200_pred_100.jsonl" # 51.25
    # json_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/our_rl_600_pred_100.jsonl" # 54.03
    # read jsonl file
    with open(json_file, "r") as f:
        lines = f.readlines()
    
    vid_dict = {}
    new_lines = []
    for line in lines:
        new_line = json.loads(line)
        if new_line['video_path'] not in vid_dict:
            vid_dict[new_line['video_path']] = line
            new_lines.append(line)
    # print(len(vid_dict.keys()))
    print(len(new_lines))
    lines = new_lines[:100]

    ious = []
    for line in lines:
        data = json.loads(line)
        pred_interval = data["timeline"]
        gt_interval = data["gt_timeline"]

        if pred_interval and gt_interval:
            iou = compute_iou(pred_interval, gt_interval)
        else:
            iou = 0.0
        ious.append(iou)

    # df["IoU"] = ious
    print(len(ious))
    print(f"Mean IoU: {sum(ious)/len(ious):.4f}")
    # df.to_csv("with_iou.csv", index=False)

if __name__ == "__main__":
    main()
