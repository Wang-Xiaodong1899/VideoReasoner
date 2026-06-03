import json

json1 = "longvideoreason_qa_5997.json"
json2 = "Video-R1-260k-filter-video-0-10000-8828.json"

# merge data
with open(json1, "r") as f:
    data1 = json.load(f)

with open(json2, "r") as f:
    data2 = json.load(f)

data = data1 + data2

with open("Video-R1-260k-filter-video-0-10000-8828-longvideoreason_qa_5997.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
