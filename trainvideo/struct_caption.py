from gpt import chat
from doubao import chat_text, chat_with_video

query = """

Please provide detailed and comprehensive captions for the following content:
1. Short Caption: Summarize the video in one detailed sentence, capturing key actions and the overall mood. 
2. Background Caption: Provide a detailed description of the background, including objects, location, weather, time, and any dynamic elements such as movements in the environment. 
3. Main Object Caption: Give a thorough description of the main subject’s actions, attributes, interactions, and movements throughout the video frames, including changes in posture, expression, or speed. 
4. Camera Caption: Describe the camera work in detail, including shot types, angles, movements, transitions, and any special effects used to enhance the video. 
5. Detailed Caption: Generate a detailed dense caption for the video. The caption should capture all visible actions, environmental details, and the overall emotional atmosphere in depth. Describe in detail the interactions between the main subjects and their environment, including subtle nuances of their movements or expressions.

Make sure to provide a vivid portrayal that is engaging, informative, and rich enough for AI to re-generate the video content. No need to provide summary content. Do not describe each frame individually. Avoid using phrases like 'first frame'. The description should be rich enough for AI to re-generate the video. Please generate the response as a Python dictionary string with keys like 'Short Caption'. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Your answers must be in English. Only provide the Python dictionary string.

"""

video_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Y6R7T.mp4"

# result_str = chat(query)
result_str = chat_with_video(query, video_path)


print(result_str)

# save to json file
import json
import ast

# 去掉 markdown 代码块标记（```python 和 ```）
if result_str.startswith("```") and result_str.endswith("```"):
    result_str = "\n".join(result_str.strip("`").split('\n')[1:])  # 去掉第一行的 ```python 和最后的 ```
    
# 解析为 Python 字典
try:
    result_dict = ast.literal_eval(result_str)
except Exception as e:
    raise ValueError("Failed to parse the result string into a Python dictionary.").with_traceback(e.__traceback__)



# # 保存为 JSON 文件
with open('struct_caption_Y6R7T.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)