from gpt import chat
from doubao import chat_text
import json
import ast
import os
from tqdm import tqdm


def pipe(video_path, event_name, times, event_idx=0):
    video_name = os.path.basename(video_path)
    save_name = video_name.split('.')[0]

    save_file = os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/event_0", f"response_event_id_{event_idx}_{save_name}.txt")

    if os.path.exists(save_file):
        print(f"Reasoning of {video_path} exists, skip...")
        return

    with open(os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1", f"struct_caption_{video_name}.json"), 'r') as f:
        struct_caption = json.load(f)
        print(f"reading caption from struct_caption_{video_name}.json")
    
    # first question
    with open(os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1", f"grounding_2QA_{video_name}.json"), 'r') as f:
        grounding_qa = json.load(f)
    
    item = grounding_qa[event_name][0] # take first qa
    ques = item["question"]
    options = item["options"]
    answer = item["answer"]

    option_dict = {}
    for option in options:
        symbol = option.split('.')[0]
        option_dict[symbol] = option.split('.')[1].strip()
    
    answer = option_dict[answer]

    prompt_template = """You are simulating human reasoning based on visual video observation. Given these inputs:
    - Video description: {video_description}
    - Question: {question}
    - Answer: {answer}
    - Event to locate: "{event_name}"
    - Event timestamps: {timestamps}

    Generate a single coherent paragraph explaining the visual reasoning process. Follow this structure exactly:
    1. Start by stating what you need to observe to answer the question
    2. Locate the event with <event_start>[start, end]<event_end> tags
    3. Indicate focused observation with <video_zoomin>
    4. Describe specific visual details you would see in that segment
    5. Connect those observations logically to the given answer

    Example response:
    "To determine 'What does the chef do after picking up the knife?', I first look for when the chef handles the kitchen knife. The key moment occurs between <event_start>[12.4, 15.2]<event_end>. After obtaining the video content of this event <video_zoomin>, I observe the chef's right hand grasping the knife handle firmly, then making precise vertical motions while keeping the knife's tip anchored against the cutting board. This visual evidence leads me to conclude 'chops vegetables' because the hand movements and resulting vegetable pieces clearly show chopping action."

    Important:
    - Phrase everything as first-person visual recollection
    - Never mention the video description - only what you "see"
    - Keep the entire response as one paragraph
    - Use precise timestamps and the required tags
    - Make the visual details specific and plausible

    Return your response as a paragraph only.
    """

    # Usage example:
    prompt = prompt_template.format(
        video_description=struct_caption,
        question=ques,
        answer=answer,
        event_name=event_name,
        timestamps=times,
    )
    # print(prompt)

    result_str = chat_text(prompt, "doubao-1-5-thinking-pro-250415")
    result_str = result_str.strip()
    # print(result_str)

    # save the string
    with open(os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/event_0", f"response_event_id_{event_idx}_{save_name}.txt"), 'w') as f:
        f.write(result_str)

def main():

    with open("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/train_filter.json", 'r') as f:
        time_query = json.load(f)
    
    count = -1
    for video_id, v in tqdm(time_query.items()):
        timestamps = v["timestamps"]
        querys = v["sentences"]

        count += 1

        caption_path = os.path.join("/mnt/bn/multimodal-datasets-hl/wangxd/Open-R1-Video-Mix/Charades_v1/", f"struct_caption_{video_id}.mp4.json")
        if not os.path.exists(caption_path):
            continue

        try:
            # TODO fix bug:
            # each query overwrite the response txt
            # for times, query in zip(timestamps, querys):
            # import pdb; pdb.set_trace()
            times, query = timestamps[0], querys[0] # 0 denote query 0
            video_path = os.path.join("/mnt/bn/wk-data-storage/wangxd/dataset/charades-dataset/Charades_v1", video_id+'.mp4')
            pipe(video_path, query, times, 0) # 0 denote query 0
            
        except Exception as e:
            print(e)
            continue

main()