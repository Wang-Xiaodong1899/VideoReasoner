import pandas as pd

df = pd.read_csv("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/videomme_long.csv")

df["question"] = df["question"].apply(lambda x: x.replace("The best answer is:", "").strip() + "\nInput a video, a question, and some options. You need to extract key elements from these visual and text information. Sort the key elements by importance. The elements at the front are more important to answer the question. Output a line of data, with each element separated by a comma. The number of key elements should not be less than 4 and no more than 10. Key elements cannot be symbols such as A/B/C/D. Only output one line of data, do not output irrelevant content.")

df.to_csv("/mnt/bn/wxd-video-understanding/wangxd/Open-R1-Video-Mix/perception_encoder/data/videomme_long_key.csv", index=False)