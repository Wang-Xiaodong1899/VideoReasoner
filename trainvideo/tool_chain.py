import json
import os
import pandas as pd
from tqdm import tqdm

with open("llava_2_3m_id2video.json", 'r') as f:
    id2video_map = json.load(f)

with open("keyframes_idx_max_128_0622_all_id_0_1800.jsonl", 'r') as f:
    keyframes_list = [json.loads(line) for line in f]

keyframes_list_dict = {}
for item in keyframes_list:
    keyframes_list_dict[item['id']] = item['keyframes']

with open("keyframes_caption_list_3B_0623.jsonl", 'r') as f:
    caption_list = [json.loads(line) for line in f]

caption_list_dict = {}
for item in caption_list:
    caption_list_dict[item['id']] = item["captions"]

# id
# captions dict

exist_ids = []
for item in caption_list:
    exist_ids.append(item["id"])

# QA data
with open("/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1/2_3_m_academic_mc_v0_1_qa_processed.json", "r") as f:
    data = json.load(f)

instruct = """I give you a question for a video and 4 options. 
I give you a detailed description of each key frame and the final answer. 
You need to help me generate a reasoning process, indicating that the key frame detection tool was called to answer this question, and then the answer was finally obtained by looking at the visual information of each key frame and reasoning.
### Question and Options
{query}
The key frames detected are {key_indices}, and the following is a detailed description of each key frame:
{key_caption}
### Answer
{answer}
Your reasoning process can be similar to:
### Example
To answer this question, I need to detect the keyframes in the video first, call the keyframe selection tool <|keyframe_selection_tool|>, and the selection result is <|keyframe_start|>[0, 124]<|keyframe_end|>. By looking at the visual content of these keyframes <|keyframes_embed|>, I analyze the details provided for each frame.

First keyframe: The image shows a living room with a couch, a coffee table, and a rug. There is a bookshelf in the background, and the room is well-lit and clean. No person is visible, and there is no indication of activity like reading, watching TV, working out, or cleaning.

Second keyframe: The image again shows the living room with a blue and black rug, a couch, a coffee table with books stacked on it, and a lamp. The room has white walls and a door. Again, no person is visible, and no specific activity is depicted.

Given that neither keyframe shows a person engaged in any of the listed activities (reading, watching TV, cleaning, or working out), but the question asks for the main activity, the most plausible inference is that the video focuses on the environment rather than a person. However, since the provided answer is C (a person doing a workout routine), it suggests that the workout equipment or setup might be implied (e.g., the rug could be for exercises, or the space is being used for workouts), even though it is not explicitly visible in the keyframes.

Thus, based on the given answer and the absence of other activities in the keyframes, the most likely correct choice is C.
### Tip
Your reasoning process must be reasonable, logical and smooth, and output the reasoning process in English.
Your task is to get the answer based on the correct path of reasoning, and the output must be strictly in the given format.
"""

video_convs_dict = {}
id_count = 0
for item in tqdm(data):
    id = item["id"]
    if id not in exist_ids:
        continue
    conversations = item["conversations"]
    conv = conversations[0]
    query = conv["value"].replace("<image>", "")
    video = item["video"]
    ans_conv = conversations[1]
    answer = ans_conv["value"]
    if video not in video_convs_dict:
        id_count = 0
        video_convs_dict[video] = {} # follow the mistake, #0624
    else:
        video_convs_dict[video][f"{id}_{id_count}"] = [query, answer]
        cur_id = f"{id}_{id_count}"
        id_count += 1
save_data = []
for video, id_dict in tqdm(video_convs_dict.items()):
    for id, qa_pair in id_dict.items():
        # id
        # take keyframes
        try:
            keyframes = keyframes_list_dict[id]
            keyframes_indexs = set()
            for kitem in keyframes:
                keyframes_indexs.add(int(kitem["key_index"]))
            keyframes_indexs = list(keyframes_indexs)
            keyframes_indexs.sort()
            unique_id = id[:-2]
            captions = caption_list_dict[unique_id]
            keyframe_captions = {}
            for frame_index in keyframes_indexs:
                keyframe_captions[f"{frame_index}"] = captions[f"{frame_index}"]
            query, answer = qa_pair
            key_captions = json.dumps(keyframe_captions)
            key_captions = "{" + key_captions + "}"
            # print(key_captions)
            # import pdb; pdb.set_trace()
            prompt = instruct.format(
                query=query,
                key_indices=keyframes_indexs,
                key_caption=key_captions,
                answer=answer,
            )
            entry = {}
            entry["id"] = id
            entry["video_path"] = os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1", id2video_map[unique_id])
            entry["question"] = prompt
            save_data.append(entry)
        except Exception as e:
            print(f" id {id} {e}")
df = pd.DataFrame(save_data)
df.to_csv('keyframes_caption_list_3B_0623_prompt.csv', index=False, encoding='utf-8')