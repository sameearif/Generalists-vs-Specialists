import pandas as pd
import ast
from prompts import *
from label_mapping import *
import re

def create_messages_classifier(task, k):
    df = pd.read_csv(f"k-shot/{task}.csv")
    cols = ["input", "label"]
    
    messages = [
        {"role": "system", "content": PROMPTS[task]},
    ]

    for i in range(k):
        input = df.iloc[i][cols[0]]
        if task == "ner-tagging" or task == "pos-tagging":
            input = highlight_word(string_to_list(input), df.iloc[i]["index"])
        messages.append({"role": "user", "content": input})
        output = ids_to_label(df.iloc[i][cols[1]], task)
        messages.append({"role": "assistant", "content": output})
    
    return messages

def create_messages_cot_classifier(task):
    df = pd.read_csv(f"k-shot/{task}.csv")
    cols = ["input", "label"]
    
    messages = [
        {"role": "system", "content": COT_PROMPTS[task]},
    ]

    for i in range(6):
        input = df.iloc[i][cols[0]]
        if task == "ner-tagging" or task == "pos-tagging":
            input = highlight_word(string_to_list(input), df.iloc[i]["index"])
        messages.append({"role": "user", "content": input})
        output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "label": ids_to_label(df.iloc[i][cols[1]], task)}
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def create_messages_translation(task, k, cot=False):
    df = pd.read_csv("k-shot/translation.csv")
    if task == "translation-en-ur":
        cols = ["english", "urdu"]
    else:
        cols = ["urdu", "english"]

    messages = [
        {"role": "system", "content": PROMPTS[task]},
    ]
    
    for i in range(k):
        input = df.iloc[i][cols[0]]
        translation = df.iloc[i][cols[1]]
        if cot:
            output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "translation": translation}
        else:
            output = {"translation": translation}
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def create_messages_summarization(k, cot=False):
    df = pd.read_csv("k-shot/summarization.csv")
    cols = ["text", "summary"]

    messages = [
        {"role": "system", "content": PROMPTS["summarization"]},
    ]
    
    for i in range(k):
        input = df.iloc[i][cols[0]]
        summary = df.iloc[i][cols[1]]
        if cot:
            output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "summary": summary}
        else:
            output = {"summary": summary}
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def create_messages_transliteration(k, cot=False):
    df = pd.read_csv("k-shot/transliteration.csv")
    cols = ["urdu", "english"]

    messages = [
        {"role": "system", "content": PROMPTS["transliteration"]},
    ]
    
    for i in range(k):
        input = df.iloc[i][cols[0]]
        transliteration = df.iloc[i][cols[1]]
        if cot:
            output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "transliteration": transliteration}
        else:
            output = {"transliteration": transliteration}
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def create_messages_paraphrase(k, cot=False):
    df = pd.read_csv("k-shot/paraphrase.csv")
    cols = ["text", "paraphrase"]

    messages = [
        {"role": "system", "content": PROMPTS["paraphrase"]},
    ]
    
    for i in range(k):
        input = df.iloc[i][cols[0]]
        paraphrase = df.iloc[i][cols[1]]
        if cot:
            output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "paraphrase": paraphrase}
        else:
            output = {"paraphrase": paraphrase}
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def create_messages_question_answering(k, cot=False):
    df = pd.read_csv("k-shot/question-answering.csv")
    cols = ["context", "question", "answer"]

    messages = [
        {"role": "system", "content": PROMPTS["question-answering"]},
    ]
    
    for i in range(k):
        input = f"Context:\n{df.iloc[i][cols[0]]}\n\nQuestion:\n{df.iloc[i][cols[1]]}"
        answer = ast.literal_eval(df.iloc[i][cols[2]])[0]
        if cot:
            output = {"cot": df.iloc[i]["cot"].replace("'", "\\'"), "answer": answer}
        else:
            output = {"answer": answer}
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": str(output)})
    
    return messages

def ids_to_label(ids, task):
    inv_map = {v: k for k, v in label_mapping[task].items()}
    return inv_map[int(ids)]

def string_to_list(x):
    x = ast.literal_eval(x)
    return x

def highlight_word(x, index):
    input_string = ""
    for j, token in enumerate(x):
        if j == index:
            input_string += f"<hl>{token}<hl> "
        elif j == len(x) - 1:
            input_string += f"{token}"
        else:
            input_string += f"{token} "
    return input_string

def check_abusive_status(text):
    not_abusive_match = re.search(r'not abusive', text)
    abusive_match = re.search(r'(?<!not )\babusive', text)
    
    if not_abusive_match and not abusive_match:
        return str({"label": "not abusive"})
    elif abusive_match and not not_abusive_match:
        return str({"label": "abusive"})
    else:
        return ""
    
def check_sarcastic_status(text):
    not_sarcastic_match = re.search(r'not sarcastic', text)
    sarcastic_match = re.search(r'(?<!not )\bsarcastic', text)
    
    if not_sarcastic_match and not sarcastic_match:
        return str({"label": "not sarcastic"})
    elif sarcastic_match and not not_sarcastic_match:
        return str({"label": "sarcastic"})
    else:
        return ""
    
def check_fake_news_status(text):
    not_fake_news_match = re.search(r'not fake news', text)
    fake_news_match = re.search(r'(?<!not )\bfake news', text)
    
    if not_fake_news_match and not fake_news_match:
        return str({"label": "not fake news"})
    elif fake_news_match and not not_fake_news_match:
        return str({"label": "fake news"})
    else:
        return ""
