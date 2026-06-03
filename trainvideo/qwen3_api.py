from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch

# 初始化 FastAPI 应用
app = FastAPI()

model_name = "/mnt/bn/multimodal-datasets-hl/wuzhirong/models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)


def process(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("content:", content)

    return content

class LLMInput(BaseModel):
    hypothesis: str

@app.post("/qwenllm")
def get_qwenllm(input: LLMInput):
    hypothesis = input.hypothesis

    output = process(hypothesis)

    return {
        "output": output
    }

# uvicorn qwen3_api:app --host 0.0.0.0 --port 8003