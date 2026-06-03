from gpt import chat
from doubao import chat_text

query = """
I am building a dataset for understanding vertical format live videos. Live videos refer to live videos conducted by users on short video platforms. This dataset contains only video content, no audio or other information.

The tasks are divided into perception tasks and reasoning tasks. Perception tasks are further divided into coarse-grained tasks and fine-grained tasks.

Coarse-grained tasks include: picture style, video theme, scene recognition, emotion recognition, and quality assessment.

Picture style: includes questions related to the style of the live broadcast type or visual style.
Video theme: includes questions related to video content, summary, and theme.
Scene recognition: includes questions related to video scenes.
Emotion recognition: includes questions related to the overall video atmosphere and character emotion recognition.
Quality assessment: includes questions such as brightness, color, dynamics, video quality assessment, or single-screen or multi-screen video comparison.
Please help me design 20 reasonable and moderately difficult seed questions for each of these 5 coarse-grained tasks. The 20 questions need to fully cover the possible problems of each task, including questions about visual content and timing, and these questions should be able to give good examples of at least 4 candidate options (you don't need to give examples of candidate options here). Your answers must be in English. Just provide a dictionary string.
"""

# result_str = chat(query)
result_str = chat_text(query, "doubao-1-5-thinking-pro-250415")


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

# 保存为 JSON 文件
with open('coarse_ques_doubao.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)