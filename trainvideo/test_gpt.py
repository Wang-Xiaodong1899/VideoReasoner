import torch
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import fire
from gpt import chat, chat_w_video
from doubao import chat_with_doubao


def main():
    # df = pd.read_csv("20250523_collect_first_42_en3.csv")
    df = pd.read_csv("去除电商0502-add25_en4.csv")
    for index, row in tqdm(df.iterrows()):
        ques = row['question_en']
        query = f"""
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{ques}
{row['A']}
{row['B']}
{row['C']}
{row['D']}
The best answer is:
        """
        # video_path = os.path.join("20250501-1k", f"{row['room_id']}_{row['create_time']}_{row['end_time']}.mp4")
        video_path = os.path.join("20250502-noEC", f"{row['room_id']}_{row['create_time']}_{row['end_time']}.mp4")
        
        # model_answer = process(query, video_path)
        # model_answer = chat_w_video(query, video_path, "gemini-2.5-pro-exp-03-25")
        model_answer = chat_with_doubao(query, video_path)
        print(model_answer)
        df.loc[index, 'video_path'] = video_path
        df.loc[index, 'model_answer'] = model_answer

    # save to csv
    # df.to_csv("20250523_collect_first_42_en3_qwen2.5vl-7b.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_qwen2.5vl-32b.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_gpt-4o-mini.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_gpt-4o-mini.csv", index=False)
    # df.to_csv("20250523_collect_first_42_en3_doubao-1-5-thinking-vision-pro-250428.csv", index=False)
    df.to_csv("去除电商0502-add25_en4_doubao-1-5-thinking-vision-pro-250428.csv", index=False)
    

main()