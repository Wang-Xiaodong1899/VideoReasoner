from gpt import chat
from doubao import chat_text, chat_with_video
import json
import ast
import os
from tqdm import tqdm


query = """

Please provide detailed and comprehensive captions for the following content:
1. Short Caption: Summarize the video in one detailed sentence, capturing key actions and the overall mood. 
2. Background Caption: Provide a detailed description of the background, including objects, location, weather, time, and any dynamic elements such as movements in the environment. 
3. Main Object Caption: Give a thorough description of the main subject’s actions, attributes, interactions, and movements throughout the video frames, including changes in posture, expression, or speed. 
4. Camera Caption: Describe the camera work in detail, including shot types, angles, movements, transitions, and any special effects used to enhance the video. 
5. Detailed Caption: Generate a detailed dense caption for the video. The caption should capture all visible actions, environmental details, and the overall emotional atmosphere in depth. Describe in detail the interactions between the main subjects and their environment, including subtle nuances of their movements or expressions.

Make sure to provide a vivid portrayal that is engaging, informative, and rich enough for AI to re-generate the video content. No need to provide summary content. Do not describe each frame individually. Avoid using phrases like 'first frame'. The description should be rich enough for AI to re-generate the video. Please generate the response as a Python dictionary string with keys like 'Short Caption'. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Your answers must be in English. Only provide the Python dictionary string.

"""

def gen_caption(video_path):

    # video_path = "/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Y6R7T.mp4"

    # result_str = chat(query)
    result_str = chat_with_video(query, video_path)

    # print(result_str)

    # 去掉 markdown 代码块标记（```python 和 ```）
    if result_str.startswith("```") and result_str.endswith("```"):
        result_str = "\n".join(result_str.strip("`").split('\n')[1:])  # 去掉第一行的 ```python 和最后的 ```
        
    # 解析为 Python 字典
    try:
        result_dict = ast.literal_eval(result_str)
    except Exception as e:
        raise ValueError("Failed to parse the result string into a Python dictionary.").with_traceback(e.__traceback__)

    save_name = os.path.basename(video_path).split('_')[0]

    # # 保存为 JSON 文件
    with open(f'/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/struct_caption_{save_name}.json', 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    return result_dict

def generate_prompt(structured_caption: dict, key_events: list[str]) -> str:
    """
    Generates a formatted prompt for question generation based on a structured video caption and key events.
    
    Args:
        structured_caption: A dictionary with keys: "Short Caption", "Background Caption", 
                          "Main Object Caption", "Camera Caption", "Detailed Caption" (values are strings).
        key_events: A list of strings describing key events in the video (e.g., ["A man picks up a red mug", 
                          "A cat jumps onto a table"]).
    
    Returns:
        A formatted prompt string.
    """
    # Convert structured caption dict to a readable JSON string with indentation
    structured_caption_str = json.dumps(structured_caption, indent=2)
    # Convert key events list to a readable string (e.g., ["Event 1", "Event 2"])
    key_events_str = "[" + ", ".join([f'"{event}"' for event in key_events]) + "]"
    
    prompt = f"""Given a structured video caption formatted as follows:
{structured_caption_str}

and a list of key events in the video:
{key_events_str}

Your task is as follows: 
For each key event, generate 2 questions. 
These questions must be answerable *only* by referencing the video content within the specific segment where the key event occurs. 
For each question, provide 4 multiple-choice options (labeled A-D) and specify the correct answer. 
Critically, to answer these questions accurately, a human must first temporally locate the corresponding key event in the video and then use the visual information from the video frames within that event’s timeframe—meaning the questions must strongly depend on the content of the key events themselves (e.g., details from the "Detailed Caption" or "Main Object Caption" tied to the event’s timing).

Please return the result as a dictionaries, the dict key is the event, the value is a list of 2 questions using the following format exactly:
{{
"key":
[
  {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "?"}},
  ...
],
"key":
[
  {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "?"}},
  ...
]
}}

"""
    
    return prompt

def qa_pipe(video_path, event_list):
    struct_caption = gen_caption(video_path)
    query = generate_prompt(struct_caption, event_list)

    result_str = chat_text(query, "doubao-1-5-thinking-pro-250415")

    # 去掉 markdown 代码块标记（```python 和 ```）
    if result_str.startswith("```") and result_str.endswith("```"):
        result_str = "\n".join(result_str.strip("`").split('\n')[1:])  # 去掉第一行的 ```python 和最后的 ```
        
    # 解析为 Python 字典
    try:
        result_dict = ast.literal_eval(result_str)
    except Exception as e:
        raise ValueError("Failed to parse the result string into a Python dictionary.").with_traceback(e.__traceback__)

    save_name = os.path.basename(video_path).split('_')[0]
    # 保存为 JSON 文件
    with open(f'/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/grounding_2QA_{save_name}.json', 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # read data from json file
    with open('/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    # train_data is a dict, get each item
    # 133 error
    # 380 error
    # 2325 error
    # 3093 error

    count = 0
    for video_id, video_data in tqdm(train_data.items()):
        if count <= 2325:
            count += 1
            continue
        timestamps = video_data["timestamps"]
        sentences = video_data["sentences"]
        video_path = os.path.join("/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1", video_id+'.mp4')
        try:
            qa_pipe(video_path, sentences)
            count += 1
        except Exception as e:
            print(e)
            print(f"{video_id} error")
            count += 1
            continue
