import json
import os
from tqdm import tqdm

def remove_duplicate_timestamps(data):
    """
    处理数据，去除完全相同的 timestamps 条目以及对应的 sentences 条目
    """
    processed_data = {}
    
    for video_id, video_data in tqdm(data.items()):

        # check video existance
        video_path = os.path.join("/mnt/bn/wk-data-storage/wangxd/dataset/activity-caption/ActivityNet_Captions/video/", video_id+'.mp4')
        if not os.path.exists(video_path):
            continue

        # 获取 timestamps 和 sentences
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        
        # 用于记录已经出现过的完整时间区间
        seen_intervals = set()
        unique_timestamps = []
        unique_sentences = []
        
        for ts, sent in zip(timestamps, sentences):
            # 将时间区间转换为元组（因为列表不可哈希，不能直接存入 set）
            interval = tuple(ts)
            if interval not in seen_intervals:
                seen_intervals.add(interval)
                unique_timestamps.append(ts)
                unique_sentences.append(sent)
        
        # 更新处理后的数据
        processed_data[video_id] = {
            "duration": video_data["duration"],
            "timestamps": unique_timestamps,
            "sentences": unique_sentences
        }
    
    return processed_data

# 示例使用
if __name__ == "__main__":
    # 假设你的 JSON 文件名为 input.json
    input_filename = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/activitynet/train.json"
    output_filename = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/activitynet/train_filter.json"
    
    # 读取 JSON 文件
    with open(input_filename, 'r') as f:
        data = json.load(f)
    
    # 处理数据
    processed_data = remove_duplicate_timestamps(data)
    
    # 写入处理后的数据到新文件
    with open(output_filename, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"处理完成，结果已保存到 {output_filename}")