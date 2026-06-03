import json

def transform_json(input_file, output_file):
    """
    将包含字典列表的 JSON 文件转换为以 problem_id 为键的字典的 JSON 文件
    
    Args:
        input_file (str): 输入 JSON 文件路径
        output_file (str): 输出 JSON 文件路径
    """
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据是否是列表
    if not isinstance(data, list):
        raise ValueError("输入 JSON 文件的根元素应该是一个列表")
    
    # 创建新的字典，以 problem_id 为键
    transformed_data = {}
    for item in data:
        if not isinstance(item, dict):
            continue  # 跳过非字典元素
        
        if 'problem_id' not in item:
            continue  # 跳过没有 problem_id 的字典
        
        problem_id = item['problem_id']
        transformed_data[problem_id] = item
    
    # 写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    input_json = "/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video.json"  # 替换为你的输入文件路径
    output_json = "/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/data/Video-R1-data/Video-R1-260k-filter-video_problem_id.json"  # 替换为你的输出文件路径
    transform_json(input_json, output_json)
    print(f"转换完成，结果已保存到 {output_json}")