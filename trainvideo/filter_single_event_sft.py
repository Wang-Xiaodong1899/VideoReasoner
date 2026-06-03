import json

# read data
with open("Video-R1-COT-165k-filter-video-one-event-judge-3k-ans.jsonl", 'r') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]
# filter data
problem_ids = []
new_data = []
for item in data:
    if item['judge'].strip() == 'No':
        new_data.append(item)
        problem_ids.append(item['problem_id'])

print(f'len of problem_ids: {len(problem_ids)}')

# read data from json
json_path = "/mnt/bn/multimodal-datasets-hl/wangxd/data/Video-R1-data/Video-R1-COT-165k-filter-video.json"

# read json
with open(json_path, 'r') as f:
    data = json.load(f)

# filter problem_id
new_data = []
for item in data:
    if item['problem_id'] in problem_ids:
        item['response'] = item['process']+item['solution']
        new_data.append(item)
        

# save to json
with open('Video-R1-COT-165k-filter-video-single-event.json', 'w') as f:
    json.dump(new_data, f, indent=4)
    