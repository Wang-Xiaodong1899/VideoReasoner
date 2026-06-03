import json

def merge_jsonl_files(input_files, output_file):
    """
    将多个JSONL文件合并为一个JSON文件
    
    参数:
        input_files (list): 要合并的JSONL文件路径列表
        output_file (str): 合并后的JSON输出文件路径
    """
    merged_data = []
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        merged_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"警告: 文件 {file_path} 中有一行无法解析为JSON: {e}")
                        continue
            print(f"已成功读取文件: {file_path}")
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 未找到，跳过该文件")
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合并完成! 结果已保存到 {output_file}")
    print(f"共合并了 {len(merged_data)} 条记录")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_files = [
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_0_400.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_400_800.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_800_1200.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_1200_1600.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_1600_2000.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_2000_2400.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_2400_2800.jsonl",
        "/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_2800_3200.jsonl",
                   ]
    output_file = '/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/VideoR1-video-keyframes_idx_max_64_0627_all_id_0_3200.json'
    
    merge_jsonl_files(input_files, output_file)