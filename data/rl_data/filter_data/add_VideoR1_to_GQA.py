import json

gqa_path = "filter_GQA_da_ground.json"

with open(gqa_path, 'r') as f:
    gqa_data = json.load(f)

videor1_path = "Video-R1-260k-filter-video-8828-0919.json"
with open(videor1_path, 'r') as f:
    videor1_data = json.load(f)

data = gqa_data + videor1_data

with open(f'filter_GQA_da_ground_VideoR1_da_{len(data)}.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
