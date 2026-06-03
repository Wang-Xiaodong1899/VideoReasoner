import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def cut_video_segments(json_path, videos_dir, output_dir):
    """
    更稳定的视频切割方案（使用重新编码）
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for video_id, info in tqdm(data.items()):
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f"警告: 视频文件 {video_path} 不存在，跳过")
            continue
        
        for i, (start, end) in enumerate(info["timestamps"], 1):
            output_path = os.path.join(output_dir, f"{video_id}_{i}.mp4")
            if os.path.exists(output_path):
                continue
            
            # 更稳定的ffmpeg命令（重新编码）
            cmd = [
                "ffmpeg",
                "-ss", str(start),       # 先定位起始点
                "-i", video_path,       # 然后输入文件
                "-to", str(end - start), # 计算持续时间
                "-c:v", "libx264",      # H.264编码
                "-preset", "fast",      # 平衡速度和质量
                "-crf", "23",           # 质量参数(18-28，越小质量越高)
                "-c:a", "aac",          # AAC音频
                "-b:a", "128k",         # 音频比特率
                "-avoid_negative_ts", "1",
                "-y",                   # 覆盖输出文件
                output_path
            ]
            
            try:
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
                print(f"成功切割: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"切割失败 {output_path}: {e.stderr.decode('utf-8')}")

if __name__ == "__main__":
    # 配置路径
    # json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json"  # 替换为你的JSON文件路径
    # videos_dir = "/mnt/bn/wxd-video-understanding/wangxd/dataset/charades-dataset/Charades_v1"  # 原始视频存放目录
    # output_dir = "/mnt/bn/wxd-video-understanding/wangxd/dataset/charades-dataset/Charades_v1_segments_2"  # 输出目录
    
    json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/"

    cut_video_segments(json_path, videos_dir, output_dir)