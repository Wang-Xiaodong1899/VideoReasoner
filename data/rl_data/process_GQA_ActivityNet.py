import json

json_path = "RL_GQA_Charade_ActivityNet_7k.json"

with open(json_path, 'r') as f:
    data = json.load(f)

save_data = []
for item in data:
    data_source = item["data_source"]
    if data_source == "Charades_v1":
        continue
    else:
        save_data.append(item)
print(len(save_data))

with open(f"RL_GQA_ActivityNet_{len(save_data)}.json", 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
