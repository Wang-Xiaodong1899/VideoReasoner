import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/src/qwen-vl-utils/src")

import torch
# from .model import VTR_Model
from pe import CLIP
from tokenizer import SimpleTokenizer
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import time

import torch
from PIL import Image
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from qwen_vl_utils import process_vision_info

import json
from tqdm import tqdm
import fire

class PE_VTR_Model():
    def __init__(self, device, model_path="/mnt/bn/multimodal-datasets-hl/llhuang/models/PE-Core-G14-448/PE-Core-G14-448.pt"):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        model_name = str(model_path).split("/")[-1].split(".")[0]
        self.clip_model = CLIP.from_config(model_name, pretrained=True,checkpoint_path=model_path,pool_type="attn").to(device)
        self.image_size = self.clip_model.image_size
        self.context_length = self.clip_model.context_length
        self.tokenizer = SimpleTokenizer(context_length=self.context_length)
        self.image_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=F.InterpolationMode.BICUBIC),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def get_text_embedding(self, text):
        text_inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs, normalize=True)
        return text_embedding 

    def get_video_embedding(self,
                            frames, return_image_embeddings=False):
        # images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
        # default frames are video tensors
        frames = frames.float() / 255.0
        images = frames.to(self.device)
        frames = self.image_transform(images) # [N, C, H, W]
        batch_size = 32
        num_frames = frames.size(0)
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch_frames = frames[i:i+batch_size]
                batch_embeddings = self.clip_model.encode_image(batch_frames, normalize=True)
                all_embeddings.append(batch_embeddings)
        image_embeddings = torch.cat(all_embeddings, dim=0)
        if return_image_embeddings:
            return image_embeddings
        video_embedding = image_embeddings.unsqueeze(0).mean(dim=1) #[1, dim]
        return video_embedding
    
# PE model
model = PE_VTR_Model(device="cuda:0")

def get_video_embeddings(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    # "max_pixels": 360 * 420,
                    "max_pixels": 720 * 480,
                    "fps": 2.0,
                }
            ]
        },
    ]

    _, video_inputs = process_vision_info(messages) # list of tensor

    video_tensors = video_inputs[0] # video tensor, N C H W [0, 1]

    video_embedding = model.get_video_embedding(video_tensors, return_image_embeddings=True)

    return video_embedding

def qwen_process(query):
    import subprocess
    message = {
        "model": "/mnt/bn/multimodal-datasets-hl/wuzhirong/models/Qwen3-8B", 
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 8192,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    command = ["curl", "http://localhost:8000/v1/chat/completions", "-H", "Content-Type: application/json",
                "-d", f"{json.dumps(message)}" ]
    print(command)
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        response = result.stdout
        print("响应内容：")
        print(response)
    else:
        print("错误信息：", result.stderr)
    try:
        data = json.loads(response)
        output = data["choices"][0]["message"]["content"]
    except json.JSONDecodeError:
        print("返回值不是有效的 JSON")
        output = ""

    print(output)
    return output

def get_query_embedding(query):
    instruct = """
Given a question about a video and candidate options, you need to summarize the core entity objects and detailed targets that need to be paid attention to in answering this question. Used for me to extract key frames from the video. Your output must only contain physical entities, which cannot be conceptually repeated and cannot be abstract concepts, sorted by the importance of answering this question, separated by commas. Answer all physical entities, detailed targets or scenes related to the question you want to answer, not abstract concepts. If you cannot get a specific target from the question and options, please return null.
"""
    prompt = instruct + query
    print(query)
    items = qwen_process(prompt)
    if items is None or not isinstance(items, str):
        ques = query.split("\n")[0]
        queries = [ques]
    else:
        queries = items.split(',')
    
    # only take five items
    queries = queries[:8]
    queries = [query.strip() for query in queries]
    print(queries)
    text_embedding = model.get_text_embedding(queries) # N, 1280
    return queries, text_embedding

def get_queries_embedding(queries):
    queries = queries[:8]
    queries = [query.strip() for query in queries]
    print(queries)
    text_embedding = model.get_text_embedding(queries) # N, 1280
    return queries, text_embedding

def search_indices(video_embedding, queries, text_embedding):
    similarity = text_embedding @ video_embedding.T # MXD DXN -> MXN

    topk = torch.topk(similarity, k=1, dim=1)  # 返回值是一个命名元组 (values, indices)

    # 每个文本对应的前4个帧索引（shape: M x 4）
    top_indices = topk.indices  # shape: (M, 4)

    # 若也需要相似度值
    top_values = topk.values  # shape: (M, 4)

    top_indices_text = []

    for i in range(top_indices.size(0)):  # 对每个文本
        for j in range(top_indices.size(1)):  # 每个文本前4帧
            frame_idx = top_indices[i, j].item()
            top_indices_text.append(
                {
                    "text": queries[i],
                    "key_index": frame_idx
                }
            )
    
    return top_indices_text

def process(query, video_path):
    
    video_embedding = get_video_embeddings(video_path)
    queries, text_embedding = get_query_embedding(query)

    return search_indices(video_embedding, queries, text_embedding)

def process_queries(queries, video_path):
    video_embedding = get_video_embeddings(video_path)
    queries, text_embedding = get_queries_embedding(queries)
    return search_indices(video_embedding, queries, text_embedding)
    

def main(start=0, end=400):
    with open("/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1/2_3_m_academic_mc_v0_1_qa_processed.json", "r") as f:
        data = json.load(f)
    video_convs_dict = {}
    id_count = 0
    for item in tqdm(data):
        id = item["id"]
        conversations = item["conversations"]
        conv = conversations[1] if len(conversations) > 1 else conversations[0] # take conv 1
        query = conv["value"].replace("<image>", "").split("\nPlease")[0]
        video = item["video"]
        if video not in video_convs_dict:
            id_count = 0
            video_convs_dict[video] = {} # forget the first qa
        else:
            video_convs_dict[video][f"{id}_{id_count}"] = query
            id_count += 1
    count = 0
    with open(f"keyframes_idx_max_64_0627_conv1_all_id_{start}_{end}.jsonl", "w") as f:
        for video, ques_id_dict in tqdm(video_convs_dict.items()):
            if count < start:
                count += 1
                continue
            if count >= end:
                break
            video_path = os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1", video)
            try:
                video_embedding = get_video_embeddings(video_path)
                print(ques_id_dict)
                for q_id, q_value in ques_id_dict.items():
                    queries, text_embedding = get_query_embedding(q_value)
                    if 'null' not in queries:
                        indice_dict = search_indices(video_embedding, queries, text_embedding)
                    else:
                        indice_dict = []
                    entry = {
                        "id": q_id,
                        "keyframes": indice_dict
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f.flush()
            except Exception as e:
                print(f"error for {video} {e}")
                continue
            count += 1

if __name__ == "__main__":
    fire.Fire(main)