curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/mnt/bn/multimodal-datasets-hl/wuzhirong/models/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'