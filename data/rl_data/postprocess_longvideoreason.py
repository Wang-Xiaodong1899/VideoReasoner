import json
import os
import pandas as pd

video_dir = "/mnt/bn/wxd-video-understanding/wangxd/data/LongVideo-Reason/longvila_videos"

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/longvideoreason/train_exist_25744.json"
with open(json_path, 'r') as f:
    longvideoreason_data = json.load(f)

problem_id = 0
save_data = []
# 0908: 0-6000
# 0909: 6000-12000
for item in longvideoreason_data[6000:18000]:
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
        # import pdb; pdb.set_trace()
        continue
    times = []
    duration = 0
    data_source = 'longvideoreason'
    save_data.append({
        "problem_id": problem_id,
        "times": times,
        "problem": problem,
        "data_type": data_type,
        "problem_type": "qa",
        "prefix": "The answer is: ",
        "options": options,
        "solution": answer,
        "path": path,
        "data_source": data_source,
        "duration": duration,
    })
    problem_id += 1
    # import pdb; pdb.set_trace()

# video_path,question,answer,seed16vl
csv_path = "data/rl_data/longvideoreason_keyitems_2000_seed16vl.csv"
df = pd.read_csv(csv_path)
# df to list of dict
df_data = df.to_dict(orient="records")

add_key=False

if add_key:
    # merge df and save_data
    for item in df_data:
        item['problem_id'] = problem_id
        video_path = item["video_path"]
        question = item["question"]
        answer = item["answer"]
        output = item["seed16vl"]
        options = question.split("\n")
        problem = options[0]
        options = options[1:-1]
        # parse a list ['A.xxx', 'B.xx'] to a dict {'A': 'xxx', 'B': 'xx'}
        # option_dict = {}
        # for option in options:
        #     option_dict[option[0]] = option[2:]

        # vid = video_path.split("/")[-1].split(".")[0]
        # import pdb; pdb.set_trace()

        times = []
        duration = 0
        data_source = 'longvideoreason'
        save_data.append({
            "problem_id": problem_id,
            "times": times,
            "problem": problem,
            "data_type": data_type,
            "problem_type": "keyitem",
            "prefix": "I want to output the key elements: ",
            "options": options,
            "solution": str(output),
            "path": video_path,
            "data_source": data_source,
            "duration": duration,
        })
        problem_id += 1

# save to json file
# with open(f"data/rl_data/longvideoreason_qa_keyitem_{len(save_data)}.json", "w") as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

with open(f"data/rl_data/longvideoreason_qa_{len(save_data)}_0909.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)