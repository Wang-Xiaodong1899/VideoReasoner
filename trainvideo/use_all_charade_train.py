import os
import pandas as pd

df = pd.read_csv('/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/query2question_seed_9k_ques.csv')

# trasfer df to json list
json_list = df.to_dict(orient='records')

save_data = []
problem_id = 0
for item in json_list:
    vid = item['vid']
    duration = item['duration']
    times = item['timestamp']
    sentence = item['sentence']
    # parse list str to list
    times = times.replace('[', '').replace(']', '').split(',')
    times = [float(time.strip()) for time in times]
    problem = item['query']
    data_type = 'video'
    options = item.get('options', [])
    problem_type = item.get('problem_type', '')
    answer = item.get('answer', '')
    path = os.path.join('/mnt/bn/wxd-video-understanding/wangxd/dataset/charades-dataset/Charades_v1', vid + '.mp4')
    data_source = 'Charades_v1'
    solution = item.get('solution', '')
    if not os.path.exists(path):
        print(path)
        continue
    save_data.append({
        'problem_id': problem_id,
        'vid': vid,
        'query': sentence,
        'times': times,
        'problem': problem,
        'data_type': data_type,
        'problem_type': problem_type,
        'options': options,
        'answer': answer,
        'path': path,
        'data_source': data_source,
        'solution': solution,
        'duration': duration
    })
    problem_id += 1

# to json file
import json
with open('/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/charades_v1_train_grpo_iou_only.json', 'w') as f:
    json.dump(save_data, f, indent=4)
