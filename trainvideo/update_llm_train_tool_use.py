import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/llm_train_tool_use.json"

with open(json_path, "r") as f:
    data = json.load(f)

save_data = []
for item in data:
    response = item["response"]
    prefix = response.split("<|keyframe_selection_tool|>")[0]
    tail = response.split("<|keyframe_selection_tool|>")[1]
    output = "<think> I want to use the keyframe selection tool <|keyframe_selection_tool|> to identify the relevant frames" + tail
    item["response"] = output
    save_data.append(item)

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/data/sft_data/keyframe_selection_tool_use_update_0708.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
