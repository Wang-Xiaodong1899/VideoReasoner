import json
import pandas as pd

json_path = "mix_sft_data.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
problem_id = 0
for item in data:
    item['problem_id'] = problem_id
    response = item['response']
    options = item['options']
    option_dict = {}
    for option in options:
        option_dict[option[0]] = option
    answer = item['answer'] if 'answer' in item else item["solution"].replace("<answer>", "").replace("</answer>", "")
    if "<|event_end|>" in response:
        item['response'] = response.split("<|event_end|>")[0].replace("<think>", "").strip() + "<|event_end|>"
    elif "<|keyframe_selection_tool|>" in response:
        continue
    else:
        item['response'] = "The answer is: " + response.split("<answer>")[-1].replace("</answer>", "")
        if item["problem_type"] == "multiple choice":
            item['response'] = "The answer is: " + option_dict[response.split("<answer>")[-1][0]]
    save_data.append(item)
    problem_id += 1

# video_path,question,answer,seed16vl
csv_path = "keyframe_sft_data_seedvl16.csv"
df = pd.read_csv(csv_path)
# df to list of dict
df_data = df.to_dict(orient="records")

# merge df and save_data
for item in df_data:
    item['problem_id'] = problem_id
    video_path = item["video_path"]
    question = item["question"]
    answer = item["answer"]
    output = item["seed16vl"]
    options = question.split("\n")
    problem = options[0]
    options = options[1:5]
    # parse a list ['A.xxx', 'B.xx'] to a dict {'A': 'xxx', 'B': 'xx'}
    option_dict = {}
    for option in options:
        option_dict[option[0]] = option[2:]

    vid = video_path.split("/")[-1].split(".")[0]
    # import pdb; pdb.set_trace()
    new_item = {
        "problem_id": problem_id,
        "vid": vid,
        "path": video_path,
        "problem": problem,
        "data_type": "video",
        "problem_type": "multiple choice",
        "options": options,
        "answer": answer,
        "response": "I want to output the key elements: " + str(output),
        "data_source": "llava_178k_2_3m",
        "keyframes": [],
    }
    save_data.append(new_item)
    problem_id += 1

# save to json file
with open("mix_sft_data_new_0720.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
