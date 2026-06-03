import json
from doubao import chat_text
from tqdm import tqdm

json_path = "/mnt/bn/multimodal-datasets-hl/wangxd/data/Video-R1-data/Video-R1-COT-165k-filter-video.json"

# read json
with open(json_path, 'r') as f:
    data = json.load(f)

query_template = """
I will give you a question about a video.
If this question only requires locating the interval of a certain "event" in the video and can be answered only by the video content of the interval (assuming that you can perceive the video content of this interval), then this question is called a "question that can be answered by single event location", please reply Yes.
If this question is about video summary, theme identification, order of event occurrence, comparison of multiple events, and other questions that cannot be answered by simply locating the video content of a certain event interval, these questions are called "questions that cannot be answered by single event location", please reply No.

Given question: {question}

You only need to reply Yes or No.
"""


# write jsonl
with open("Video-R1-COT-165k-filter-video-one-event-3k-10k-judge-ans.jsonl", 'w') as f:
    for item in tqdm(data[3000:10000]):
        problem_id = item["problem_id"]
        problem = item["problem"]
        query = query_template.format(question=problem)
        answer = chat_text(query)
        print(answer)
        f.write(json.dumps({"problem_id": problem_id, "problem": problem, "judge": answer}) + "\n")
        f.flush()

