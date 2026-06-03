import json

# data_path = "train_event_id_0_3k_query_time_latest.json"
data_path = "train_event_id_0_3k_query_time_0705.json"

with open(data_path, 'r') as f:
    data = json.load(f)

ok_data = []
for item in data:
    response = item['response']
    count = response.count("<|event_start|>")
    flag = 1
    if count > 1:
        print(item['problem_id'])
        print(response)
        print(count)
        print('====================')
        flag = 0
    count = response.count("<|event_end|>")
    if count > 1:
        print(item['problem_id'])
        print(response)
        print(count)
        print('====================')
        flag = 0
    count = response.count("<|video_zoomin|>")
    if count > 1:
        print(item['problem_id'])
        print(response)
        print(count)
        print('====================')
        flag = 0
    count = response.count("<|segment_pad|>")
    if count > 1:
        print(item['problem_id'])
        print(response)
        print(count)
        print('====================')
        flag = 0
    if flag:
        ok_data.append(item)
# with open("train_event_id_0_3k_query_time_latest_check.json", "w") as f:
#     json.dump(ok_data, f, indent=4, ensure_ascii=False)
with open("train_event_id_0_3k_query_time_0705_check.json", "w") as f:
    json.dump(ok_data, f, indent=4, ensure_ascii=False)
    
