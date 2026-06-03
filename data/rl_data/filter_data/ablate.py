import json

json_path = "filter_GQA_da_ground.json"

with open(json_path, "r") as f:
    data = json.load(f)

ground_data = []
qa_data = []
for item in data:
    if item["problem_type"] == "grounding":
        ground_data.append(item)
    else:
        qa_data.append(item)

with open("filter_GQA_da_ablate.json", "w") as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=4)

with open("filter_GQA_ground_ablate.json", "w") as f:
    json.dump(ground_data, f, ensure_ascii=False, indent=4)


