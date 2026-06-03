import json
import os

video_dir = "/mnt/bn/wxd-video-understanding/wangxd/data/LongVideo-Reason/longvila_videos"

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/longvideoreason/train_exist_25744.json"
with open(json_path, 'r') as f:
    longvideoreason_data = json.load(f)

prompt = "Input a video, question and options. You need to extract key elements from these visual and text information. Key elements cannot be too similar or repeated. Output a line of data, with each element separated by a comma. There must be no less than 4 key elements and no more than 16 key elements. Key elements cannot be symbols such as A/B/C/D. You only need to output one line of data, and do not output irrelevant content."

problem_id = 0
save_data = []
for item in longvideoreason_data[:2000]:
    # key items
    # accuracy
    problem = item["problem"]
    data_type = item["data_type"]
    videos = item["videos"]
    path = os.path.join(video_dir, videos)
    answer = item["answer"].replace("<answer>", "").replace("</answer>", "").strip()
    options = problem.split("\n")
    options = [option.strip() for option in options if option.strip() != ""][1:]
    option_dict = {}
    for option in options:
        option_dict[option[0]] = option
    try:
        answer = option_dict[answer]
    except:
        print(answer, option_dict)
        continue
    question = problem.split("\n")[0].strip()
    problem = question + "\n" + "\n".join(options) + "\n" + prompt
    save_data.append({
        "video_path": path,
        "question": problem,
        "answer": answer,
    })

# save to csv file
import pandas as pd
df = pd.DataFrame(save_data)
df.to_csv("longvideoreason_keyitems_2000.csv", index=False)


