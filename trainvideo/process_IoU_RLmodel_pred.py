import json


with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/stage_1_IoU_RLModel_pred_all.json") as f:
    data = json.load(f)
save_data = []
for item in data:
    iou_reason = item["iou_reason"]
    times = item["times"] # [s, e]
    think = iou_reason + f"{times}" + "<|event_end|>. Focusing on this segment <|video_zoomin|><|video_pad|>,"
    item["think"] = think
    save_data.append(item)

with open("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/stage_1_IoU_RLModel_pred_GT_time.json", "w") as f:
    json.dump(save_data, f, indent=4)
