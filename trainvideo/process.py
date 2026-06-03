import pandas as pd
from tqdm import tqdm

from gpt import translate_with_gpt, chat

# df = pd.read_csv("20250523_collect_first_42_en2.csv")
# df = pd.read_csv("去除电商0502-add25_en2.csv")
df = pd.read_csv("去除电商0502-add25_en3.csv")

def distractor(question, answer):
    query = f"""
You are an expert multiple-choice question generator. Given a question and its unique correct answer, generate three plausible but incorrect distractors. Combine them with the correct answer to form four options (A/B/C/D), shuffle their order randomly, and output:

A. Option A content
B. Option B content
C. Option C content
D. Option D content
correct_letter: The letter (A/B/C/D) of the correct option

Question: {question}
Answer: {answer}

Your output:

    """

    return chat(query)

# 遍历df
for index, row in tqdm(df.iterrows()):

    # question_en = translate_with_gpt(row['question'])
    # answer_en = translate_with_gpt(row['answer'])

    # # translate to english
    # df.loc[index, 'question_en'] = question_en.strip()
    # df.loc[index, 'answer_en'] = answer_en.strip()

    # ques = row['question_en']
    # ans = row['answer_en']
    # output = distractor(ques, ans)
    # print(output)
    # df.loc[index, 'output'] = output

    # process
    output = row['output']
    items = output.split('\n')
    df.loc[index, 'A'] = items[0].strip()
    df.loc[index, 'B'] = items[1].strip()
    df.loc[index, 'C'] = items[2].strip()
    df.loc[index, 'D'] = items[3].strip()
    df.loc[index, 'correct'] = items[4].split('correct_letter:')[-1].strip()
    # pass


# only save question answer columns
select = ["user_id", "room_id", "task", "question", "question_en", "answer", "answer_en", "A", "B", "C", "D", "correct", "output", "date", "hour", "create_time", "end_time"]
# select = ["user_id", "room_id", "task", "question", "question_en", "answer", "answer_en", "date", "hour", "create_time", "end_time"]

# select = ["user_id", "room_id", "task", "question", "question_en", "answer", "answer_en", "output", "date", "hour", "create_time", "end_time"]


df = df[select]

df.to_csv("去除电商0502-add25_en4.csv", index=False)