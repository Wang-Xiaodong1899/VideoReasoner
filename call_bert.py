import requests

url = "http://localhost:8002/bertscore"
data = {
    "hypothesis": "The quick brown fox jumps over the lazy dog.",
    "reference": "A fast brown fox leaps over a lazy dog."
}

response = requests.post(url, json=data)
print(response.json()["f1"])