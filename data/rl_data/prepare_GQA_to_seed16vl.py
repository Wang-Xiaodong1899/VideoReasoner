import json
import os


json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/filter_data/filter_GQA_qa_ground.json"
with open(json_path, 'r') as f:
    GQA_data = json.load(f)

test_data = []
for item in GQA_data:
    if item["problem_type"] == "qa":
        test_data.append(item)

prompt = "Input a video, a question, and some options. You need to extract key elements from these visual and text information. Sort the key elements by importance. The elements at the front are more important to answer the question. Output a line of data, with each element separated by a comma. The number of key elements should not be less than 4 and no more than 10. Key elements cannot be symbols such as A/B/C/D. Only output one line of data, do not output irrelevant content."

problem_id = 0
save_data = []
for item in test_data:
    # key items
    # accuracy
    solution = item["solution"]
    answer = solution
    path = item["path"]
    problem = item["problem"]
    problem = problem + "\n" + prompt
    save_data.append({
        "video_path": path,
        "question": problem,
        "answer": answer,
    })

# save to csv file
import pandas as pd
df = pd.DataFrame(save_data)
df.to_csv("GQA_keyitems.csv", index=False)


