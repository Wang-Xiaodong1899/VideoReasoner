import json
import pandas as pd

# json_path = "mix_sft_data.json"

# with open(json_path, "r") as f:
#     data = json.load(f)

# save_data = []
# for item in data:
#     respone = item["response"]
#     if "<|keyframe_selection_tool|>" in respone:
#         save_data.append(item)

# # save to json
# with open("keyframe_sft_data.json", "w") as f:
#     json.dump(save_data, f, indent=2)

json_path = "keyframe_sft_data.json"

with open(json_path, "r") as f:
    data = json.load(f)

prompt = """
Input a video, question and options. You need to extract key elements from these visual and text information. Key elements cannot be too similar or repeated. Output a line of data, with each element separated by a comma. There must be no less than 4 key elements and no more than 16 key elements. Key elements cannot be symbols such as A/B/C/D. You only need to output one line of data, and do not output irrelevant content.
"""

save_data = []
for item in data:
    video_path = item["path"]
    problem = item["problem"]
    answer = item["answer"]
    options = item["options"] # list
    question = problem + "\n" + "\n".join(options)
    question = question + prompt
    save_data.append({
        "video_path": video_path,
        "question": question,
        "answer": answer
    })

# save_data to csv file
data = pd.DataFrame(save_data)
data.to_csv("keyframe_sft_data.csv", index=False)