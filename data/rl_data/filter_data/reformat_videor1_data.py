import json
import os

# {
#         "problem_id": 2,
#         "problem": "What appears on the screen in Russian during the missile's ascent?",
#         "data_type": "video",
#         "problem_type": "multiple choice",
#         "options": [ 
#             "A. A YouTube subscription notification",
#             "B. A military command",
#             "C. A warning message",
#             "D. A weather update"
#         ],
#         "solution": "<answer>A</answer>",
#         "path": "./LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4",
#         "data_source": "LLaVA-Video-178K/30_60_s_youtube_v0_1"
#     },

# process to
# {
#         "problem_id": 0,
#         "times": [],
#         "problem": "What is the man's primary goal in the workshop, as inferred from the video?  \nA. Creating a prop replica of Ash's chainsaw hand from \"Evil Dead\"  \nB. Building a custom storage container for mixed nuts  \nC. Installing a ventilation system using pipes and fans  \nD. Crafting a DIY household appliance  \n",
#         "data_type": "video",
#         "problem_type": "qa",
#         "prefix": "The answer is: ",
#         "options": [
#             "A. Creating a prop replica of Ash's chainsaw hand from \"Evil Dead\"",
#             "B. Building a custom storage container for mixed nuts",
#             "C. Installing a ventilation system using pipes and fans",
#             "D. Crafting a DIY household appliance"
#         ],
#         "solution": "A. Creating a prop replica of Ash's chainsaw hand from \"Evil Dead\"",
#         "path": "/mnt/bn/wxd-video-understanding/wangxd/data/LongVideo-Reason/longvila_videos/longvila_videos/5nKz1hzvSqs.mp4",
#         "data_source": "longvideoreason",
#         "duration": 0
#     },
json_path = "/mnt/bn/wxd-video-understanding/wangxd/data/Video-R1-data/Video-R1-260k-filter-video.json"

video_dir = "/mnt/bn/wxd-video-understanding/wangxd/data/Video-R1-data"


with open(json_path, "r") as f:
    data = json.load(f)

data = data[:10000]
save_data = []
for item in data:
    item["problem_id"] = int(item["problem_id"])
    item["times"] = []
    if item["problem_type"] != "multiple choice":
        continue
    item["problem_type"] = "qa"
    item["prefix"] = "The answer is: "
    options = item["options"]
    item["problem"] = item["problem"] + "\n" + "\n".join(options)
    # solution = item["solution"].replace("<answer>", "").replace("</answer>", "")
    # option_dict = {}
    # for option in options:
    #     option_dict[option[0]] = option
    # try:
    #     item["solution"] = option_dict[solution[0]]
    # except:
    #     print(solution)
    #     continue
    item["duration"] = 0
    item["path"] = os.path.join(video_dir, item["path"][2:])
    item["data_source"] = "Video-R1"
    # import pdb; pdb.set_trace()
    save_data.append(item)

new_json_path = f"Video-R1-260k-filter-video-{len(save_data)}-0919.json"

with open(new_json_path, "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
