import pandas as pd
import os
from glob import glob

# 设置 CSV 文件所在的文件夹路径
folder_path = '.'  # 替换为你的路径

# 使用 glob 找到所有 CSV 文件
csv_files = glob(os.path.join(folder_path, '*_ques.csv'))

# 读取所有 CSV 文件并合并为一个 DataFrame
df_all = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# 保存合并后的结果
output_path = os.path.join(folder_path, 'query2question_seed_9k_ques.csv')
df_all.to_csv(output_path, index=False)

print(f'合并完成，保存到: {output_path}')
