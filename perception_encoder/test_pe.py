import torch
# from .model import VTR_Model
import pe
from tokenizer import SimpleTokenizer
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import time

import torch
from PIL import Image
import os

from qwen_vl_utils import process_vision_info

class PE_VTR_Model():
    def __init__(self, device, model_path="/mnt/bn/multimodal-datasets-hl/llhuang/models/PE-Core-G14-448/PE-Core-G14-448.pt"):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        model_name = str(model_path).split("/")[-1].split(".")[0]
        self.clip_model = pe.CLIP.from_config(model_name, pretrained=True,checkpoint_path=model_path,pool_type="attn").to(device)
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
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(frames, normalize=True) #[N, dim]
        if return_image_embeddings:
            return image_embeddings
        video_embedding = image_embeddings.unsqueeze(0).mean(dim=1) #[1, dim]
        return video_embedding
    

model = PE_VTR_Model(device="cuda:0")

query = "Which direction does the person with intricately braided hair walk towards at the end of the video?"

options = ["Towards the back of the salon",
            "Towards the entrance of the salon",
            "Towards the stylist's station",
            "Towards the mall area"]
queries = options + [query]

queries = ["person with intricately braided hair", "salon", "entrance of the salon"]

queries = ["stacks of magazines or books", "window", "mat", "side table", "couch"]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_academic_v0_1/academic_source/activitynet/v_JOBSEatasv4.mp4",
                # "max_pixels": 360 * 420,
                "max_pixels": 720 * 480,
                "fps": 2.0,
            }
        ]
    },
]

def single_text():
    s = time.time()

    _, video_inputs = process_vision_info(messages) # list of tensor

    print(f"process_vision_info time: {time.time()-s} s")

    video_tensors = video_inputs[0] # video tensor, N C H W [0, 1]

    s = time.time()

    video_embedding = model.get_video_embedding(video_tensors, return_image_embeddings=True)

    text_embedding = model.get_text_embedding(queries)

    print(f"text_embedding shape: {text_embedding.shape}")

    similarity = text_embedding @ video_embedding.T # 1XD DXN -> 1XN

    print(f"similarity time: {time.time()-s} s")

    # single text
    similarity = similarity.squeeze(0).detach().cpu()  # shape: (N,)

    topk = torch.topk(similarity, k=8)
    top_indices = topk.indices  # shape: (8,)

    top_frames = video_tensors[top_indices]  # shape: (8, C, H, W)

    save_dir = './top_similar_frames'
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(top_frames):
        frame = frame.detach().cpu()

        frame = frame / 255

        img = F.to_pil_image(frame)  # 输入要求是 (C, H, W)，数值 [0, 1]

        img.save(os.path.join(save_dir, f"top_{i+1}.jpg"))

def multi_text():
    s = time.time()

    _, video_inputs = process_vision_info(messages) # list of tensor

    print(f"process_vision_info time: {time.time()-s} s")

    video_tensors = video_inputs[0] # video tensor, N C H W [0, 1]

    s = time.time()

    video_embedding = model.get_video_embedding(video_tensors, return_image_embeddings=True)

    text_embedding = model.get_text_embedding(queries)

    print(f"text_embedding shape: {text_embedding.shape}")

    similarity = text_embedding @ video_embedding.T # MXD DXN -> MXN

    topk = torch.topk(similarity, k=4, dim=1)  # 返回值是一个命名元组 (values, indices)

    # 每个文本对应的前4个帧索引（shape: M x 4）
    top_indices = topk.indices  # shape: (M, 4)

    # 若也需要相似度值
    top_values = topk.values  # shape: (M, 4)

    for i in range(top_indices.size(0)):
        print(f"文本 {i} 最相似的帧索引: {top_indices[i].tolist()}，相似度值: {top_values[i].tolist()}")

    save_dir = './top_frames_by_text'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(top_indices.size(0)):  # 对每个文本
        for j in range(top_indices.size(1)):  # 每个文本前4帧
            frame_idx = top_indices[i, j].item()
            frame = video_tensors[frame_idx].detach().cpu() / 255
            img = F.to_pil_image(frame)
            img.save(os.path.join(save_dir, f"text_{i}_top_{j+1}_frame_{frame_idx}.jpg"))

multi_text()