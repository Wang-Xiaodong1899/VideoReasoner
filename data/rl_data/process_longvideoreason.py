import json
import os

video_dir = "/mnt/bn/wxd-video-understanding/wangxd/data/LongVideo-Reason/longvila_videos"

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/longvideoreason/train_exist_25744.json"
with open(json_path, 'r') as f:
    longvideoreason_data = json.load(f)

problem_id = 0
save_data = []
for item in longvideoreason_data:
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
add_keyitem = False
if add_keyitem:
    for item in longvideoreason_data:
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
        # import pdb; pdb.set_trace()
        try:
            answer = option_dict[answer]
        except:
            print(answer, option_dict)
            continue
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
            "solution": answer,
            "path": path,
            "data_source": data_source,
            "duration": duration,
        })
        problem_id += 1

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/RL_GQA_ActivityNet_5232.json"

with open(json_path, 'r') as f:
    gqa_activity_data = json.load(f)

add_GQA_temp = False

if add_GQA_temp:
    for item in gqa_activity_data:
        # key items
        # accuracy
        problem = item["problem"]
        data_type = item["data_type"]
        path = item["path"]
        options = item["options"]
        problem = problem + "\n".join(options)
        option_dict = {}
        for option in options:
            option_dict[option[0]] = option
        answer = item["answer"]
        answer = option_dict[answer]
        times = item["times"]
        duration = item["duration"]
        data_source = 'GQA'
        save_data.append({
            "problem_id": problem_id,
            "times": times,
            "problem": problem,
            "data_type": data_type,
            "problem_type": "grounding",
            "prefix": "I want to locate the key event in the video. ",
            "options": options,
            "solution": answer,
            "path": path,
            "data_source": data_source,
            "duration": duration,
        })
        problem_id += 1
        # import pdb; pdb.set_trace()

for item in gqa_activity_data:
    # key items
    # accuracy
    problem = item["problem"]
    data_type = item["data_type"]
    path = item["path"]
    options = item["options"]
    problem = problem + "\n".join(options)
    option_dict = {}
    for option in options:
        option_dict[option[0]] = option
    answer = item["answer"]
    answer = option_dict[answer]
    times = item["times"]
    duration = item["duration"]
    data_source = 'GQA'
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

# save data
# new_json_path = f"GQA_longvideoreason_ground_items_qa_{len(save_data)}.json"
# with open(new_json_path, 'w') as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# new_json_path = f"GQA_longvideoreason_ground_qa_{len(save_data)}.json"
# with open(new_json_path, 'w') as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# TODO
# add charades 18k data
json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/rl_data/charades_find_query_train_grpo_iou_only.json"

with open(json_path, 'r') as f:
    charades_data = json.load(f)

for item in charades_data:
    save_data.append({
        "problem_id": problem_id,
        "times": item['times'],
        "problem": item['problem'],
        "data_type": "video",
        "problem_type": "grounding",
        "prefix": "I want to locate the key event in the video. ",
        "options": item['options'],
        "solution": item['solution'],
        "path": path,
        "data_source": item['data_source'],
        "duration": item['duration'],
    })
    problem_id += 1

# new_json_path = f"GQA_ActivityNet_charades_longvideoreason_ground_qa_{len(save_data)}.json"
# with open(new_json_path, 'w') as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

# new_json_path = f"GQA_ActivityNet_charades_longvideoreason_ground_qa_key_{len(save_data)}.json"
# with open(new_json_path, 'w') as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)

new_json_path = f"charades_longvideoreason_ground_qa_{len(save_data)}.json"
with open(new_json_path, 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)