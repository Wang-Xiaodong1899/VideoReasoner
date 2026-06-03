import requests

url = "http://localhost:8003/qwenllm"
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

question = """
Where are the stacks of magazines or books located?
A. Next to the window
B. On the mat
C. Under the side table
D. On the couch
"""

prompt = instruct + question
data = {
    "hypothesis": prompt,
}

response = requests.post(url, json=data)
print(response)
print(response.json()["output"])