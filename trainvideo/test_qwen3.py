from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/mnt/bn/multimodal-datasets-hl/wuzhirong/models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # comment this line if on V100 GPUs
    device_map="auto",
)

print("model loaded")

# prepare the model input
instruct = """
Given a question about a video and candidate options, you need to summarize the core entity objects and detailed targets that need to be paid attention to in answering this question. Used for me to extract key frames from the video. Your output must only contain physical entities, which cannot be conceptually repeated and cannot be abstract concepts, sorted by the importance of answering this question, separated by commas. Answer all physical entities, detailed targets or scenes related to the question you want to answer, not abstract concepts. If you cannot get a specific target from the question and options, please return null.
"""
question = """
Question:
Which direction does the person with intricately braided hair walk towards at the end of the video?
Candidates:
A. Towards the back of the salon
B. Towards the entrance of the salon
C. Towards the stylist's station
D. Towards the mall area
"""

prompt = instruct + question


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

print("thinking content:", thinking_content)
print("content:", content)
