import json

def merge_json_lists(file1_path, file2_path, output_path):
    """
    合并两个包含列表的JSON文件
    
    参数:
        file1_path (str): 第一个JSON文件路径
        file2_path (str): 第二个JSON文件路径
        output_path (str): 合并后的输出文件路径
    """
    try:
        # 读取第一个JSON文件
        with open(file1_path, 'r', encoding='utf-8') as f1:
            list1 = json.load(f1)
            if not isinstance(list1, list):
                raise ValueError("第一个文件的内容不是列表")
        
        # 读取第二个JSON文件
        with open(file2_path, 'r', encoding='utf-8') as f2:
            list2 = json.load(f2)
            if not isinstance(list2, list):
                raise ValueError("第二个文件的内容不是列表")
        
        # 合并两个列表
        merged_list = list1[:2000] + list2

        print(len(merged_list))
        
        # 写入合并后的列表到新文件
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(merged_list, out_file, ensure_ascii=False, indent=4)
            
        print(f"成功合并文件，结果已保存到 {output_path}")
    
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except ValueError as e:
        print(f"值错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# 使用示例
if __name__ == "__main__":
    file1 = "train_event_id_0_3k_query_time_latest_check.json"  # 第一个JSON文件路径
    file2 = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/trainvideo/Video-R1-COT-165k-filter-video-single-event.json"  # 第二个JSON文件路径
    output = "train_event_id_0_2k_query_time_latest_add_900_single_event.json"  # 合并后的输出文件路径
    
    merge_json_lists(file1, file2, output)