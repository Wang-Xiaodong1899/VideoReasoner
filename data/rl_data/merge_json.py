import json

# json_1 = "stage_1_IoU_RLModel_pred_GT_time_ratio.json"
# json_2 = "RL_GQA_single_segment_data.json"

# with open(json_1, 'r') as f:
#     data1 = json.load(f)

# with open(json_2, 'r') as f:
#     data2 = json.load(f)

# data = data1 + data2

# with open("RL_GQA_Charade_ActivityNet_7k.json", 'w') as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)


json_1 = "charades_v1_train_grpo_iou_only.json"
json_2 = "charades_find_query_train.json"

with open(json_1, 'r') as f:
    data1 = json.load(f)

with open(json_2, 'r') as f:
    data2 = json.load(f)

data = data1 + data2

with open("charades_find_query_train_grpo_iou_only.json", 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)