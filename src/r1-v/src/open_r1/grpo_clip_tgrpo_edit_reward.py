# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# from bert_score import score as bertscore


from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModifiedClip
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


from transformers import TrainerCallback
from deepspeed.utils import safe_get_local_grad

class ZeROGradCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if state.is_local_process_zero:
            print("=== ZeRO local gradients ===")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    grad = safe_get_local_grad(p)
                    print(f"{n:60}  has_local_grad={grad is not None}")
                else:
                    print(f"{n:60}  has_local_grad={False}")
            print("============================")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "iou"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    # add use_std
    use_std: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using reward_std"},
    )
    # add use_length_norm
    use_length_norm: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using reward length norm"},
    )



def iou_reward(completions, **kwargs):
    def extract_pred_time(text):
        pattern = r"<think>.*?<\|event_start\|>(.*?)<\|event_end\|>.*?"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            times = match.group(1)
            times = times.strip()[1:-1]
            times = times.split(',')
            times = [float(time) for time in times]
            return times
        else:
            return []

    question_type = kwargs['problem_type'][0]
    local_path = kwargs['path'][0]

    gt_times = kwargs['times'][0]

    duration = kwargs['duration'][0]

    gt_times = [float(time) / duration for time in gt_times]

    # frames
    # gt_times = kwargs['frames'][0]

    # print(f"gt_times: {gt_times}")
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content in contents:
        try:
            pred_times = extract_pred_time(content)
            gt_ans = gt_times

            if len(pred_times) == 0:
                reward = 0.0
            else:
                pred_s, pred_e = pred_times
                gt_s, gt_e = gt_times

                intersection = max(0, min(pred_e, gt_e) - max(pred_s, gt_s))
                union = max(pred_e, gt_e) - min(pred_s, gt_s)
                if union > 0:
                    iou = intersection / union
                reward = iou
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
            pred_times = [0., 0.]
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Gt_ans: {gt_times}\n")
                f.write(f"pred_times: {pred_times}\n")
                f.write(f"local_path: {local_path}\n")
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    local_path = kwargs['path'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip()[0] == gt_ans.strip()[0] else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"local_path: {local_path}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern = r"<think>.*?<\|event_start\|>.*?<\|event_end\|>.*?"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else -1.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "iou": iou_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )
SYSTEM_PROMPT = ""

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    VIDEO_QUESTION_TEMPLATE = (
        "Give you a video. Please think carefully about your reasoning strategy before answering the question.\n"
        "There are three possible reasoning modes:\n"
        "1. **Event localization**: If understanding the video requires identifying a key event (e.g., due to unclear or complex dynamics), first locate the event using the format: <|event_start|> [start_ratio, end_ratio] <|event_end|>, where the ratios are floats between 0 and 1 indicating the relative position in the video.\n"
        "   You can then use <|video_zoomin|> to inspect the event content more closely.\n"
        "2. **Keyframe selection**: If identifying specific visual moments is more helpful, use the keyframe selection tool by inserting <|keyframe_selection_tool|> and provide the result in the format: <|keyframe_start|>[list_of_frame_indices]<|keyframe_end|>. You can then reason based on the visual content using <|keyframes_embed|>.\n"
        "3. **Direct reasoning**: If the video is already clear enough, you may proceed directly without locating events or selecting keyframes.\n"
        "Please begin with your chosen reasoning path and present your thought process and final answer using the following tags:\n"
        "<think> your reasoning process here </think> <answer> your answer here </answer>\n"
        "Question: {Question}\n"
    )
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        QUESTION_TEMPLATE = VIDEO_QUESTION_TEMPLATE
        
        system_message = "You are a very smart multimodal assistant that can understand videos. "

        msg ={
            "messages": [
                   {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}]
                    },
                   {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": example['path'],
                            "max_pixels": 60*28*28, # 128*28*28, #150*28*28,
                            "max_frames": 80 #64 #32
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question)
                        }
                        ]
                }
                ]
            }
        # import pdb; pdb.set_trace()
        
        return msg
    
    dataset = dataset.map(make_conversation_image_and_video)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModifiedClip
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        # callbacks=[ZeROGradCallback()]
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
