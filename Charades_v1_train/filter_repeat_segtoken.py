import json
from tqdm import tqdm

# read json file
# with open('train_event_id_0_3k_query_time_latest.json', 'r') as f:
#     data = json.load(f)
with open('train_event_id_0_3k_query_time_0705.json', 'r') as f:
    data = json.load(f)

ok_data = []

for item in tqdm(data):
    response = item["response"]
    if response.count("") == 2:
        print(f"error")
    else:
        ok_data.append(item)

# save to json file
# with open("train_event_id_0_3k_query_time_latest_check.json", "w") as f:
#     json.dump(ok_data, f, indent=4, ensure_ascii=False)
with open("train_event_id_0_3k_query_time_0705_check.json", "w") as f:
    json.dump(ok_data, f, indent=4, ensure_ascii=False)