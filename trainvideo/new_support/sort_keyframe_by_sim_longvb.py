import json

# lvbench确实全部都是768帧
# json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/lvbench_f768_seed16vl_segment_query8_seg32_topk1_index.json"

# 0920

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/0924_longvideobench_f768_ours_segment_query8_seg32_topk1_index.json"


with open(json_path, "r") as f:
    data = json.load(f)

# total_frames = 768 # 好像不一定总是768帧

save_data = []
for item in data:
    total_frames = item["total_frames"]
    idx_list = item["idx_list"]
    idx_list = sorted(idx_list, key=lambda x: x["similarity_score"], reverse=True)
    item["idx_list"] = idx_list
    detail_list =  item["idx_list"]
    save_detail_list = []
    for detail in detail_list:
        start_frame = detail["start_frame"]
        end_frame = detail["end_frame"]
        start_pos = start_frame / (total_frames-1)
        end_pos = end_frame / (total_frames-1)
        detail["time_ratio"] = [start_pos, end_pos]
        save_detail_list.append(detail)
    item["idx_list"] = save_detail_list
    save_data.append(item)

with open(json_path.split(".json")[0] + "_sort.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
