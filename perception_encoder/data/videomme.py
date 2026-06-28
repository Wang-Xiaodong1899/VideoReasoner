import json
import os
import pandas as pd
import re

class VideoMME:
    def __init__(self,
                 video_dir,
                 anno_path,
                 subset = None,
                 post_prompt="The best answer is:",
                 ):
        self.video_dir = video_dir
        self.raw_anno_path = anno_path
        self.subset = subset
        self.subtile_dir = "/mnt/bn/wxd/wuzhirong/hf_cache/videomme/subtitle"

        df = pd.read_parquet(self.raw_anno_path)
        if self.subset == "long w/o subs" or self.subset == "long w subs":
            data = df[df['duration']=='long'].to_json(orient="records")
        elif self.subset == "short": # default w/o subs
            data = df[df['duration']=='short'].to_json(orient="records")
        elif self.subset == "medium": # default w/o subs
            data = df[df['duration']=='medium'].to_json(orient="records")
        elif self.subset == "long": # default w/o subs
            data = df[df['duration']=='long'].to_json(orient="records")
        else:
            data = df.to_json(orient="records")
        data = json.loads(data)

        # our pred event
        csv_file = f"/mnt/wxd/wangxd/eval/Open-R1-Video-V1/eval/benchmarks/videomme-{self.subset}-ques-event.csv"

        df = pd.read_csv(csv_file)

        # df to json
        df = df.to_dict(orient='records')

        path2interval = []
        think_result = []
        for item in df:
            output = item['qwenvl25_7b_mix_fix_data_temp']
            match = re.search(r"\[(\d+\.\d+),\s*(\d+\.\d+)\]", output)
            if match:
                num1, num2 = match.groups()
                interval = [float(num1), float(num2)]
                prefix = output.split('[')[0]
                tail = output.split(']')[1]
                think_result.append(prefix+ f"<|event_start|>[{interval[0]:.2f}, {interval[1]:.2f}]<|event_end|>. " + 'Focusing on this segment <|video_zoomin|><|segment_pad|>,')
            else:
                print("未找到匹配的 [num1, num2] 格式")
                # interval = [0., 0.]
                print(output)
                # raise ValueError
                interval = None
                think_result.append(output)
            
            path2interval.append(interval)
        
        print(path2interval[0], len(path2interval))
        # import pdb; pdb.set_trace()
        skip_keyframe = False

        # if self.subset == "short":
        #     keyframe_data_path = "/mnt/wxd/wangxd/VideoReasoner/videomme_short_f128_keyframe_index.json"
        # elif self.subset == "medium":
        #     keyframe_data_path = "/mnt/wxd/wangxd/VideoReasoner/videomme_medium_f768_keyframe_index.json"
        # else:
        #     # report not support error
        #     # raise ValueError(f"Not support subset {self.subset}")
        #     # warning
        #     print(f"Warning: Not support subset {self.subset}")
        #     skip_keyframe = True
        if self.subset == "short":
            keyframe_data_path = "/mnt/wxd/wangxd/VideoReasoner/videomme_short_f768_our-EGRPO_keyframe_index.json"
        elif self.subset == "medium":
            keyframe_data_path = "/mnt/wxd/wangxd/VideoReasoner/videomme_medium_f768_our-EGRPO_keyframe_index.json"
        else:
            keyframe_data_path = "/mnt/wxd/wangxd/VideoReasoner/videomme_long_f768_our-EGRPO_keyframe_index.json"

        if not skip_keyframe:
            with open(keyframe_data_path, 'r') as f:
                keyframe_data = json.load(f)
        
        # ques, idx_list (dict)


        self.rows = []
        row_id = 0
        for i in range(len(data)):
            video_id, ques, opts, ans = data[i]['videoID'], data[i]['question'],data[i]['options'], data[i]['answer']
            # option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
            option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
            option = "\n".join([f"{opt}" for i, opt in enumerate(opts)])
            question = ques + "\n" + option
            full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

            if self.subset == "long w subs" or self.subset == "full w subs":
                subtitle_path = os.path.join(self.subtile_dir, data[i]["videoID"] + ".srt")
                if os.path.exists(subtitle_path):
                    subtitle = open(subtitle_path).readlines()
                else:
                    subtitle = []
                subtitles_prompt = "This video's subtitles are listed below: \n"
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
                full_prompt = subtitles_prompt + subtitle_text + "\n" + full_prompt
            if not skip_keyframe:
                sorted_dict = sorted(keyframe_data[i]['idx_list'], key=lambda x: x["key_index"])
            else:
                sorted_dict = []

            self.rows.append({
                "id" : row_id,
                "video_path" : video_path,
                "ques" : ques,
                "opts" : [opt[2:] for opt in opts],
                "gt_ans" : ans,
                "full_prompt" : full_prompt,
                "duration_type" : data[i]['duration'],
                "query": question,
                "pe_frame_s" : path2interval[i],
                "think_result" : think_result[i],
                "keyframe_data" : sorted_dict,
            })
            row_id+=1

    def metrics(self, result_path):
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['duration_type']
        res = {}
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                if item['ans'][0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                res[task_type][1]+=1
        m = []
        total, count = 0,0
        for k, v in res.items():
            count+=v[0]
            total+=v[1]
            m.append({
                "type" : k,
                "true" : v[0],
                "total" : v[1],
                "accuracy" : v[0]/v[1] if v[1]>0 else 0,
            })
        if self.subset == "full w/o subs" or self.subset == "full w subs":
            m.append({
                "type" : "overall",
                "true" : count,
                "total" : total,
                "accuracy" : count/total if total>0 else 0,
            })
        return m

    def metrics_box(self, result_path):
        def extract_conclusion_answer(text):
            pattern = "oxed\{(.*?)\}" # TODO debug
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['duration_type']
        res = {}
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                answer_extract = extract_conclusion_answer(item['ans'])
                print(f"extracted: {answer_extract}")
                if len(answer_extract)==0:
                    answer_extract = " "
                if answer_extract[0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                res[task_type][1]+=1
        m = []
        total, count = 0,0
        for k, v in res.items():
            count+=v[0]
            total+=v[1]
            m.append({
                "type" : k,
                "true" : v[0],
                "total" : v[1],
                "accuracy" : v[0]/v[1] if v[1]>0 else 0,
            })
        if self.subset == "full w/o subs" or self.subset == "full w subs":
            m.append({
                "type" : "overall",
                "true" : count,
                "total" : total,
                "accuracy" : count/total if total>0 else 0,
            })
        return m

    def metrics_think(self, result_path):
        def extract_conclusion_answer(text):
            pattern = r"<answer>(.*?)</answer>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['duration_type']
        res = {}
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                answer_extract = extract_conclusion_answer(item['ans'])
                print(f"extracted: {answer_extract}")
                answer_extract = answer_extract.strip()
                if len(answer_extract)==0:
                    answer_extract = " "
                if answer_extract[0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                res[task_type][1]+=1
        m = []
        total, count = 0,0
        for k, v in res.items():
            count+=v[0]
            total+=v[1]
            m.append({
                "type" : k,
                "true" : v[0],
                "total" : v[1],
                "accuracy" : v[0]/v[1] if v[1]>0 else 0,
            })
        if self.subset == "full w/o subs" or self.subset == "full w subs":
            m.append({
                "type" : "overall",
                "true" : count,
                "total" : total,
                "accuracy" : count/total if total>0 else 0,
            })
        return m

    