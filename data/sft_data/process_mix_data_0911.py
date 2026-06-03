import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/mix_sft_data_new_0720.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
for item in data:
    response = item["response"]
    # skip direct answer query
    if "The answer is:" in response:
        continue
    save_data.append(item)

video_rft_path = "/mnt/bn/wxd-video-understanding/wangxd/Long-RL/data/VideoRFT/VideoRFT-Video-CoT-Data_15840.json"

with open(video_rft_path, "r") as f:
    video_rft_data = json.load(f)

problem_id = 10000
for item in video_rft_data[:1500]:
    problem_type = item["problem_type"]
    if problem_type != "multiple choice":
        continue
    problem = item["problem"]
    res = problem.split("\n")
    problem = res[0]
    options = res[1:]
    response = item["response"]
    response = "Analysis and answer: " + response.replace("<think>", "").replace("</think>", " ")
    save_data.append({
        "problem_id": problem_id,
        "problem": problem,
        "data_type": "video",
        "problem_type": "multiple choice",
        "options": options,
        "solution": item["solution"],
        "path": item["path"],
        "data_source": "LLaVA-Video-178K",
        "response": response,
    })
    problem_id += 1

print(len(save_data))
import pdb; pdb.set_trace()

new_json_path = f"/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/mix_sft_data_new_{len(save_data)}_0911.json"
with open(new_json_path, "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)