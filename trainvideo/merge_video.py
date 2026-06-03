from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import os

def extract_frames(video_path, output_path, num_frames=64, thumbnail_width=100):
    # 加载视频文件
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # 计算等间隔的时间点
    times = np.linspace(0, duration, num_frames, endpoint=False)
    
    # 提取帧并调整大小
    frames = []
    for t in times:
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        
        # 计算新高度以保持宽高比
        aspect_ratio = img.height / img.width
        thumbnail_height = int(thumbnail_width * aspect_ratio)
        
        # 调整大小
        img = img.resize((thumbnail_width, thumbnail_height))
        frames.append(img)
    
    # 计算拼接后的大图尺寸
    total_width = thumbnail_width * num_frames
    max_height = max([img.height for img in frames])
    
    # 创建新图像
    combined = Image.new('RGB', (total_width, max_height))
    
    # 拼接所有缩略图
    x_offset = 0
    for img in frames:
        combined.paste(img, (x_offset, 0))
        x_offset += thumbnail_width
    
    # 保存结果
    combined.save(output_path)
    print(f"成功保存拼接图像到: {output_path}")

# 使用示例
# video_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/lokFoo_QD8c.mp4"  # 替换为你的视频文件路径
# output_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/lokFoo_QD8c_merge.jpg"
# extract_frames(video_path, output_path, num_frames=32, thumbnail_width=100)

video_path = "/mnt/bn/wk-data-storage/wuzhirong/datasets/LVBench/all_videos/Za2Z_JRxCuk.mp4"  # 替换为你的视频文件路径
output_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/assets/Za2Z_JRxCuk_merge.jpg"
extract_frames(video_path, output_path, num_frames=32, thumbnail_width=200)