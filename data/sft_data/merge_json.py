import json

path = "train_event_id_0_1k_query_time_latest_add_900_single_event_update_0708_no_pad.json"

with open(path, "r") as f:
    data = json.load(f)

path1 = "keyframe_selection_tool_use_update_0708_no_pad.json"

with open(path1, "r") as f:
    data1 = json.load(f)

merge_data = data + data1

# remove error time
for idx, item in enumerate(merge_data):
    if "times" in item:
        times = item['times']
        duration = item['duration']
        if times[-1] > duration:
            item['times'][-1] = duration
        if times[0] < 0:
            item['times'][0] = 0.0
        points = item['points']
        if points[-1] > 1:
            item['points'][-1] = 1.0
            item['end_point'] = 1.0
        if points[0] < 0:
            item['points'][0] = 0.0
            item['start_point'] = 0.0
    merge_data[idx] = item

with open("mix_sft_data.json", 'w') as f:
    json.dump(merge_data, f, indent=4, ensure_ascii=False)