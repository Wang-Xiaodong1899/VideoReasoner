import base64
import cv2
import os
from openai import OpenAI

import time


client = OpenAI(
    base_url="https://api.ai-gaochao.cn/v1/",
    api_key="xxx",
)

def translate_with_gpt(text):
    completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful translator"
                        },
                        {
                            "role": "user",
                            "content": f"Translate this into English:\n{text} \n Only output English text.\n"
                        },
                    ]
                )
    return completion.choices[0].message.content.strip()

def chat(query, model_name="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": query
            },
        ]
    )
    return completion.choices[0].message.content.strip()

def chat_w_video(query, video_path, model_name="gpt-4o"):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    completion = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"These are frames from a video. {query}"
                        )
                    },
                    *[
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{frame}"
                        }
                        for frame in base64Frames[0::25]
                    ]
                ]
            }
        ],
    )
    return completion.choices[0].message.content.strip()


# import requests

# print(translate_with_gpt("视频中有几个棉花？"))
# print(completion.model)
