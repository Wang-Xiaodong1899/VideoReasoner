# bert_score_api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from bert_score import BERTScorer
import torch

# 初始化 FastAPI 应用
app = FastAPI()

# 加载 scorer（推荐在 GPU 上加载）
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)

# 请求输入的数据格式
class ScoreInput(BaseModel):
    hypothesis: str
    reference: str

@app.post("/bertscore")
def get_bertscore(input: ScoreInput):
    hypothesis = [input.hypothesis]
    reference = [input.reference]

    P, R, F1 = scorer.score(hypothesis, reference)

    return {
        "precision": round(P[0].item(), 4),
        "recall": round(R[0].item(), 4),
        "f1": round(F1[0].item(), 4)
    }

# uvicorn bert_api:app --host 0.0.0.0 --port 8002