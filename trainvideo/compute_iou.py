import re
import pandas as pd
import ast
import fire

def extract_time_interval(text):
    # 匹配类似 "0.60 - 5.10" 或 "18.00 to 32.00"
    pattern = r'(\d+(?:\.\d+)?)\s*(?:-|to|,)\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, text)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return [start, end] if start < end else [end, start]
    return None

def extract_time_range(s):
    s = str(s)
    match = re.findall(r"\d+\.?\d*", s)
    return [float(m) for m in match] if len(match) == 2 else None

def compute_iou(pred, gt):
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0

def main(csv_path, key):
    df = pd.read_csv(csv_path)
    # df = pd.read_csv("/mnt/bn/wxd-video-understanding/wangxd/repo/wzr_eval/xiaodong/scripts/test_charades_val_all_for_qwen25vl_output.csv")  # 替换为你的文件路径
    # df = pd.read_csv("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/test_charades_val_for_qwen25vl_100_res.csv")
    # df = pd.read_csv("/mnt/bn/wxd-video-understanding/wangxd/repo/wzr_eval/xiaodong/scripts/test_charades_val_all_for_seed15vl_output.csv")  # 替换为你的文件路径
    
    ious = []
    thresholds = [0.3, 0.5, 0.7]
    counts = {threshold: 0 for threshold in thresholds}  # 计数每个阈值下的成功次数
    for idx, row in df.iterrows():
        duration = row['duration']
        try:
            pred_interval = extract_time_interval(row[key])
            pred_interval = [pred_interval[0] * duration, pred_interval[1] * duration]
        except Exception:
            pred_interval = None
        # pred_interval = extract_time_range(row["seed15vl"])
        try:
            gt_interval = ast.literal_eval(row["timestamp"])  # 例如 [0.6, 5.1]
        except Exception:
            gt_interval = None
        if pred_interval and gt_interval:
            iou = compute_iou(pred_interval, gt_interval)
        else:
            iou = 0.0
        ious.append(iou)
        for threshold in thresholds:
            if iou >= threshold:
                counts[threshold] += 1

    total = len(ious)
    print(f"Total samples: {total}")
    print(f"Mean IoU: {sum(ious)/total:.4f}")

    for threshold in thresholds:
        accuracy = counts[threshold] / total
        print(f"IoU@{threshold}: {accuracy:.4f} ({counts[threshold]}/{total})")

if __name__ == "__main__":
    fire.Fire(main)
