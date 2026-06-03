import os
from volcenginesdkarkruntime import Ark

# client = Ark(api_key="4166326a-436b-4576-8790-9331a204182e")
client = Ark(api_key="0559e2f5-2935-43a8-b6dc-04414763b707")

import base64

def video_to_base64(file_path):
    """
    将视频文件转换为 Base64 编码（带 MIME 类型前缀）
    
    :param file_path: 视频文件路径（如：'video.mp4'）
    :return: Base64 编码字符串，格式如：data:video/mp4;base64,<编码>
    """
    # 1. 提取视频格式（从文件扩展名）
    video_format = file_path.split('.')[-1].lower()
    mime_type = f"video/{video_format}"

    # 2. 读取视频文件的二进制数据
    with open(file_path, 'rb') as video_file:
        video_data = video_file.read()

    # 3. 将二进制数据编码为 Base64
    base64_encoded = base64.b64encode(video_data).decode('utf-8')

    # 4. 拼接完整格式
    result = f"data:{mime_type};base64,{base64_encoded}"
    return result

def chat_with_video(query, video_path, model="doubao-1-5-thinking-vision-pro-250428", fps=2):
    video_url = video_to_base64(video_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "content":[
                    {
                        "video_url": {"url":video_url, "fps": fps},
                        "type": "video_url"
                    },
                    {
                        "text": query, 
                        "type": "text"
                    },
                ],
                "role":"user"
            }
        ],
        max_tokens=16000,
    )
    # print(resp.choices[0].message.content)
    return resp.choices[0].message.content


def chat_text(query, model="doubao-1-5-thinking-pro-250415"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"content":[{"text":query,"type":"text"}],"role":"user"}],
        max_tokens=16000
    )
    # print(resp.choices[0].message.content)
    return resp.choices[0].message.content

# print(chat_text("你好"))
# print(chat_text("你好", "doubao-1-5-thinking-pro-250415"))

# doubao-1-5-thinking-pro-250415
# doubao-1-5-thinking-vision-pro-250428
# doubao-1-5-vision-pro-250328
