import json
import os
from tqdm import tqdm

data_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_no_1k3_times.json"
with open(data_path, 'r') as f:
    data = json.load(f)

qa_root = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/"
video_root = "/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1/"

with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json", 'r') as f:
    time_query = json.load(f)

write_data = []
problem_id = 0
for item in tqdm(data):
    vid = item['vid']
    qa2_path = os.path.join(qa_root, f"grounding_2QA_{vid}.mp4.json")
    # each query has two qa

    anno_item = time_query[vid]

    timestamps = anno_item['timestamps'] # len
    querys = anno_item['sentences'] # len

    query = querys[0] # first query
    timestamp = timestamps[0] # first time
    with open(qa2_path, 'r') as f:
        qa_data = json.load(f)
    query_2_qa_list = qa_data[query]
    query_qa = query_2_qa_list[0]
    new_item = {
        "problem_id": problem_id,
        "vid": vid,
        "query": query,
        "times": timestamp,
        "problem": query_qa['question'],
        "data_type": "video",
        "problem_type": "multiple choice",
        "options": query_qa['options'],
        "answer": query_qa['answer'],
        "path": os.path.join(video_root, f"{vid}.mp4"),
        "data_source": "Charades_v1",
    }
    problem_id += 1
    write_data.append(new_item)

with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_0_no_1k3_4grpo.json", 'w') as f:
    json.dump(write_data, f, indent=4, ensure_ascii=False)
