import pandas as pd
import os

# 读取 CSV 文件
input_file = '/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/query2question_seed.csv'  # 替换为你的文件名
df = pd.read_csv(input_file)

# 拆分成 10 份
n_splits = 10
chunk_size = len(df) // n_splits
remainder = len(df) % n_splits

# 创建输出文件夹
output_dir = 'split_csv'
os.makedirs(os.path.join("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo", output_dir), exist_ok=True)

start = 0
for i in range(n_splits):
    # 处理余数，让前几块多一行
    extra = 1 if i < remainder else 0
    end = start + chunk_size + extra
    chunk = df.iloc[start:end]
    chunk.to_csv(f'/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/trainvideo/{output_dir}/query2question_seed_part_{i+1}.csv', index=False)
    start = end
