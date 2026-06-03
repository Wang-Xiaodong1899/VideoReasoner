import json

# read jsonl file
with open("failed_reasoning.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

ok_data = []
failed_data = []
for item in data:
    judge = item["judge"]
    # if judge == "Yes":
    #     ok_data.append(item)
    if judge == "No":
        failed_data.append(item)

# write ok data to json file
# with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_yes_1k7.json", "w") as f:
#     json.dump(ok_data, f, indent=4, ensure_ascii=False)
with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1_train/" + "train_event_id_0_no_1k3.json", "w") as f:
    json.dump(failed_data, f, indent=4, ensure_ascii=False)
