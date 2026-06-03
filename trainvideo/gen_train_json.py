import os
import json
from tqdm import tqdm

# read all response*.txt in "folder"
# folder = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/"
folder = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/event_0"
files = os.listdir(folder)
video_ids = []
for file in files:
    if "response_event_id_0" in file:
        video_ids.append(file.split("_")[-1].split('.')[0])

print('video ids', len(video_ids))

# import pdb; pdb.set_trace()

# write json file
data = []

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json", 'r') as f:
    time_query = json.load(f)

problem_id = 0
for id in tqdm(video_ids):

    try:
        video_path = os.path.join("/mnt/bn/wxd-video-understanding/wangxd/dataset/charades-dataset/Charades_v1", id+'.mp4')
        segment_path = os.path.join("/mnt/bn/wxd-video-understanding/wangxd/dataset/charades-dataset/Charades_v1_segments_2", id+'_1.mp4')

        anno_item = time_query[id]
        duration = anno_item['duration']
        timestamps = anno_item['timestamps']
        querys = anno_item['sentences']

        # pick_timestamp = timestamps[-1]
        # pick_query = querys[-1]
        pick_timestamp = timestamps[0]
        pick_query = querys[0]


        # query
        # read grounding_2QA_{id}.json
        with open(os.path.join("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/", 'grounding_2QA_'+id+'.mp4'+'.json'), 'r') as f:
            grounding_qa = json.load(f)
        
        item = grounding_qa[pick_query][0] # take first qa
        ques = item["question"]
        options = item["options"]
        answer = item["answer"]
        

        # long response
        # read from response_{id}.txt
        with open(os.path.join(folder, 'response_event_id_0_'+id+'.txt'), 'r') as f:
            response = f.read()

        response = response.replace("<event_start>", "<|event_start|>")
        response = response.replace("<event_end>", "<|event_end|>")
        response = response.replace("<video_zoomin>", "<|video_zoomin|><|segment_pad|>")
        
        full_answer = next((opt for opt in options if opt.startswith(answer + ".")), answer)

        response = "<think> " + response + "</think>" + "<answer> " + full_answer + " </answer>"

        item = {
            "problem_id": problem_id,
            "vid": id,
            "duration": duration,
            "query": pick_query,
            "times": pick_timestamp,
            "problem": ques,
            "data_type": "video",
            "problem_type": "multiple choice",
            "options": options,
            "answer": answer,
            "path": video_path,
            "segment_path": segment_path,
            "response": response,
            "data_source": "Charades_v1"
        }
        problem_id += 1

        data.append(item)
        
    except Exception as e:
        print(f"{id} {e}")
        continue

# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_3k.json", "w") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_3k_query_time.json", "w") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

# with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_3k_query_time_latest.json", "w") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_3k_query_time_0705.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)