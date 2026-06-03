import os
import pandas as pd
import json
import random

def get_mid_segment_frame_indices(duration, segments, total_frames=80):
    """
    获取多个segment中间时刻对应的帧的下标
    
    参数:
        duration: 视频总时长(秒)
        segments: 包含多个segment的列表，每个segment是[start, end]表示起止时间(秒)
        total_frames: 视频均匀采样的总帧数，默认为128
        
    返回:
        包含每个segment中间时刻对应帧下标的列表
    """
    frame_indices = []
    
    for segment in segments:
        start, end = segment
        mid_time = (start + end) / 2  # 计算中间时刻
        
        # 计算中间时刻对应的帧下标
        # 均匀采样，所以帧间隔是duration/total_frames
        frame_index = int((mid_time / duration) * total_frames)
        
        # 确保下标在有效范围内[0, total_frames-1]
        frame_index = max(0, min(frame_index, total_frames - 1))
        
        frame_indices.append(frame_index)
    
    return frame_indices

def randomize_options(a0, a1, a2, a3, a4, answer):
    options = [a0, a1, a2, a3, a4]
    
    correct_index = options.index(answer)
    
    random.shuffle(options)
    
    new_correct_index = options.index(answer)
    
    labels = ['A', 'B', 'C', 'D', 'E']
    
    options = [f"{label}. {option}" for label, option in zip(labels, options)]

    correct = f"{labels[new_correct_index]}"
    
    return options, correct


df_meta = pd.read_csv("/mnt/bn/wxd-video-understanding/wangxd/data/NExT-GQA/datasets/nextgqa/val.csv")

# dataframe to dict list
meta_list = df_meta.to_dict(orient='records')

json_path = "/mnt/bn/wxd-video-understanding/wangxd/data/NExT-GQA/datasets/nextgqa/gsub_val.json"

with open(json_path, 'r') as f:
    anno_data = json.load(f)

id2path_path = "/mnt/bn/wxd-video-understanding/wangxd/data/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"

with open(id2path_path, 'r') as f:
    id2path = json.load(f)

# single segment data
single_segment_data = []

problem_id = 10000
for item in meta_list:
    video_id = item['video_id']
    question = item['question']
    answer = item['answer']
    qid = item['qid']
    a0 = item['a0']
    a1 = item['a1']
    a2 = item['a2']
    a3 = item['a3']
    a4 = item['a4']

    video_path = os.path.join("/mnt/bn/wxd-video-understanding/wangxd/data/NExT-GQA/datasets/NExTVideo", id2path[str(video_id)]+".mp4")

    video_anno = anno_data[str(video_id)]
    duration = video_anno['duration']
    location = video_anno['location']
    timestamp = location[str(qid)] # [[start, end]]
    if len(timestamp) == 1:
        continue
    elif len(timestamp) > 1:
        problem_id += 1
        # start, end = timestamp[0]
        # points = [start/duration, end/duration]
        # start_point = points[0]
        # end_point = points[1]
        options, correct = randomize_options(a0, a1, a2, a3, a4, answer)
        keyframes = get_mid_segment_frame_indices(duration, timestamp)
        # think = f"<think>I want to locate the key event in the video. To determine {question}, I need to observe the segment. The relevant event occurs between a proportion of <|event_start|>[{start_point:.2f}, {end_point:.2f}]<|event_end|>. Focusing on this segment <|video_zoomin|><|segment_pad|>,"
        pad_str = "<|vision_start|><|image_pad|><|vision_end|>" * len(keyframes)
        think = f"<think> I want to use the keyframe selection tool <|keyframe_selection_tool|> to identify the relevant frames, and the selection result is <|keyframe_start|>{keyframes}<|keyframe_end|>. By looking at the visual content of these keyframes <|keyframes_embed|>{pad_str}, I analyze the details provided for each frame.\n\n"
        single_segment_data.append(
            {
            'problem_id': problem_id,
            'vid': str(video_id),
            'query': "",
            'times': timestamp,
            'problem': question,
            'data_type': "video",
            'problem_type': "multiple choice",
            'options': options,
            'answer': correct,
            'path': video_path,
            'data_source': "GQA",
            'solution': f"<answer>{correct}</answer>",
            'duration': duration,
            'keyframes': keyframes,
            'think': think
        })
    

# save single_segment_data
with open("RL_GQA_multi_segment_data.json", 'w') as f:
    json.dump(single_segment_data, f, indent=4, ensure_ascii=False)


