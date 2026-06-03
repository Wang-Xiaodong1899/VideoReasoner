import json
import csv

# 输入JSON文件和输出CSV文件路径
json_file = 'trainvideo/test_charades_val_for_ours_sft.json'
csv_file = 'trainvideo/test_charades_val_for_ours_sft.json.csv'

# 读取JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 确保数据是列表形式
if isinstance(data, dict):
    data = [data]

# 获取所有可能的字段名（表头）
fieldnames = set()
for item in data:
    fieldnames.update(item.keys())

# 写入CSV文件
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"转换完成，CSV文件已保存为: {csv_file}")