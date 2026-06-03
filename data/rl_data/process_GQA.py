import os
import pandas as pd
import json
import random

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
        problem_id += 1
        start, end = timestamp[0]
        points = [start/duration, end/duration]
        start_point = points[0]
        end_point = points[1]
        options, correct = randomize_options(a0, a1, a2, a3, a4, answer)
        think = f"<think>I want to locate the key event in the video. To determine {question}, I need to observe the segment. The relevant event occurs between a proportion of <|event_start|>[{start_point:.2f}, {end_point:.2f}]<|event_end|>. Focusing on this segment <|video_zoomin|><|segment_pad|>,"
        single_segment_data.append(
            {
            'problem_id': problem_id,
            'vid': str(video_id),
            'query': "",
            'times': [start, end],
            'problem': question,
            'data_type': "video",
            'problem_type': "multiple choice",
            'options': options,
            'answer': correct,
            'path': video_path,
            'data_source': "GQA",
            'solution': f"<answer>{correct}</answer>",
            'duration': duration,
            'think': think,
            'points': [start/duration, end/duration],
        })
    

# save single_segment_data
with open("RL_GQA_single_segment_data.json", 'w') as f:
    json.dump(single_segment_data, f, indent=4, ensure_ascii=False)


