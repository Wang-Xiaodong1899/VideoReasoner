import requests
import os

def call_doubao_chat_api(user_message):
    url = "https://ark-cn-beijing.bytedance.net/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "doubao-seed-1-6-flash-250615",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }
        ],
        "thinking": {
            "type": "disabled"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def extract_assistant_message(response_json):
    print(response_json)
    for item in response_json.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return None


def extract_assistant_message(response_json):
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None

list1 = [
    "winter coat", 
    "study desk", 
    "electric kettle", 
    "bluetooth speaker", 
    "head"
]

# List 2 (Longer - 10 items with semantic matches)
list2 = [
    "puffy jacket",          # matches "winter coat"
    "writing desk",          # matches "study desk"  
    "water heater",          # matches "electric kettle"
    "wireless speaker",      # matches "bluetooth speaker"
    "jogging sneakers",      # matches "running shoes"
    "coffee mug",            # new item
    "bookshelf",             # new item
    "air fryer",             # new item
    "table lamp",            # new item
    "gym bag"                # new item
]

# 使用示例
input = f"""
You are an expert at semantic entity matching. Analyze two lists of everyday items and output the entities from list1 that have semantically equivalent matches in list2 as a single comma-separated line.

Rules:
1. A "match" can be:
   - Exact (e.g., "green apple" ↔ "Granny Smith apple")
   - Synonym/abbreviation (e.g., "TV" ↔ "television")
   - Clear contextual reference (e.g., "mountain bike" ↔ "off-road bicycle")
2. Ignore minor spelling/capitalization differences.
3. Only include list1 items with verifiable matches in list2.

Example Lists:
list1 = ["cell phone", "dining chair", "green apple", "TV", "bike"]
list2 = ["smartphone", "kitchen table chair", "Granny Smith apple", "LED television", "wireless earbuds"]

Required Output:
"cell phone, dining chair, green apple, TV"

Now process the following lists:
list1 = {list1}
list2 = {list2}
"""
response = call_doubao_chat_api(input)
message = extract_assistant_message(response)
items = [item.strip().strip("'") for item in message.split(',')]
print(items)
print(len(items))
acc = len(items)/len(list1)
recall = len(items)/len(list2)
print(acc, recall)
# get f1 score
f1 = 2 * acc * recall / (acc + recall)
print(f"F1 score: {f1:.4f}")
