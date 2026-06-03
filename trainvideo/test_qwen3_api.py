import openai

openai.api_key = "EMPTY"  # vLLM 默认不需要 API key
openai.api_base = "http://localhost:8002/v1"

response = openai.ChatCompletion.create(
    model="/mnt/bn/multimodal-datasets-hl/wuzhirong/models/Qwen3-8B",
    messages=[
        {"role": "user", "content": "请介绍一下你自己"},
    ],
    temperature=0,
    max_tokens=1024,
)

output = response["choices"][0]["message"]["content"].split("</think>")[-1].strip()
print(output)
