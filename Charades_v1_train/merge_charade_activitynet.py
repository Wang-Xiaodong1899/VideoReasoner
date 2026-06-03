import json
import random

def merge_and_shuffle_json(file1, file2, output_file):
    # 读取第一个JSON文件
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 读取第二个JSON文件
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 合并两个JSON数据
    merged_data = data1 + data2

    # edit the problem_id in data2
    offset = len(data1)

    for idx in range(len(data2)):
        data2[idx]["problem_id"] += offset
    
    # 打乱顺序
    random.shuffle(merged_data)

    print(f"len merged_data: {len(merged_data)}")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合并并打乱后的数据已保存到 {output_file}")

# 使用示例
file_1 = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/Charades_v1_train/train_event_id_1_qa_01_grpo_solution.json"
file_2 = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/activitynet/train_grpo_solution_0704.json"
output_file = "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/activitynet/train_grpo_solution_Charades_v1_activitynet_0704.json"
merge_and_shuffle_json(file_1, file_2, output_file)