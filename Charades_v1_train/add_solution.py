import json

file1_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_1_qa_0_grpo.json"

with open(file1_path, 'r', encoding='utf-8') as f1:
    list1 = json.load(f1)

new_data = []
for item in list1:
    answer = item["answer"]
    item["solution"] = "<answer>"+answer+"</answer>"
    new_data.append(item)

file_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_1_qa_0_grpo_solution.json"
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)