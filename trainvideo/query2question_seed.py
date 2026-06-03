import json
import os
import pandas as pd


json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json"
with open(json_path, "r") as f:
    data = json.load(f)

keys = data.keys()

prompt = """I give you an input event, and you need to help me ask a detailed question about this event. For example, asking about the details of the person, the details of the action, the properties of the object, etc., which require locating the event and carefully checking it before answering.

For example:

input: person takes a cup out the fridge.

output: What color is the cup that this person takes from the fridge?

The input is:

{sentence}

Your response is:
"""

save_data = []
for key in keys:
    item = data[key]
    duration = item["duration"]
    timestamps = item["timestamps"]
    sentences = item["sentences"]
    vid = key
    for idx, sen in enumerate(sentences):
        save_data.append({
            "vid": vid,
            "duration": duration,
            "timestamp": timestamps[idx],
            "sentence": sen,
            "video_path": "",
            "question": prompt.format(sentence=sen)
        })

# write save_data to csv file
df = pd.DataFrame(save_data)
df.to_csv("query2question_seed.csv", index=False)