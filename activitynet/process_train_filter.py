import json

with open("train_filter.json", 'r') as f:
    data = json.load(f)

save_dict = {}

max_duration = 0
mean_duration = 0
for k, item in data.items():
    timestamps = item['timestamps']
    duration = item['duration']
    sentences = item['sentences']
    sentences = [sen.strip() for sen in sentences]
    save_dict[k] = {
        'timestamps': timestamps,
        'duration': duration,
        'sentences': sentences
    }
    max_duration = max(max_duration, duration)
    mean_duration += duration

mean_duration = mean_duration / len(data)
print(f"max_duration: {max_duration}, mean_duration: {mean_duration}")

# save save_dict
with open("train_filter.json", 'w') as f:
    json.dump(save_dict, f, indent=4)