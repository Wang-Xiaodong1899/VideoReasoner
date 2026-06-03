from gpt import chat
from doubao import chat_text
import json
import ast

with open("struct_caption_Y6R7T.json", 'r') as f:
    struct_caption = json.load(f)


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


event_list = [
                "person start playing on their phone.",
                "person pouring it into a glass.",
            ]

query = generate_prompt(struct_caption, event_list)

print(query)

print("-----------------------")

# result_str = chat(query)
result_str = chat_text(query, "doubao-1-5-thinking-pro-250415")


print(result_str)

# save to json file


# 去掉 markdown 代码块标记（```python 和 ```）
if result_str.startswith("```") and result_str.endswith("```"):
    result_str = "\n".join(result_str.strip("`").split('\n')[1:])  # 去掉第一行的 ```python 和最后的 ```
    
# 解析为 Python 字典
try:
    result_dict = ast.literal_eval(result_str)
except Exception as e:
    raise ValueError("Failed to parse the result string into a Python dictionary.").with_traceback(e.__traceback__)

# 保存为 JSON 文件
with open('grounding_2QA_Y6R7T.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)