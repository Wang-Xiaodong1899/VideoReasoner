
import os
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("20250523_collect_first_42_en3.csv")
for index, row in tqdm(df.iterrows()):
    video_url = f"{row['room_id']}_{row['create_time']}_{row['end_time']}.mp4"
    # df.loc[index, 'tos_url'] = os.system(f"curl http://[fdbd:dc02:2b:b3c::20]:61221/get_tos_url/{video_url}")
    df.loc[index, 'tos_url'] = f"https://tosv.byted.org/obj/webcast-content-slice/wzr_webcast_videos/{video_url}"
df.to_csv("20250523_collect_first_42_en3_tos.csv", index=False)