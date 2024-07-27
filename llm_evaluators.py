from openai import OpenAI
from together import Together
import anthropic
from prompts import *
from datasets import load_dataset
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os
import ast
import re
from tqdm import tqdm

class GPT():
    def __init__(self, task):
        self.client = OpenAI(
                organization='org-OWe4x5PWbYh2yljPFA9odbCD',
            )
        self.task = task

    def forward(self, x):
        messages = [
            {"role": "system", "content": EVALUATION_PROMT[self.task]},
            {"role": "user", "content": f"input: {x[0]}\n\noutput: {x[1]}"}
        ]
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content
    
class Claude():
    def __init__(self, task):
        self.client = anthropic.Anthropic(
            api_key="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",
        )
        self.task = task

    def forward(self, x):
        messages = [
            {"role": "user", "content": f"input: {x[0]}\n\noutput: {x[1]}"}
        ]
        completion = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=CLAUDE_EVALUATION_PROMT[self.task],
            messages=messages
        )
        return completion.content[0].text
    
class LLaMA():
    def __init__(self, task):
        self.client = Together()
        self.task = task
    
    def forward(self, x):
        messages = [
            {"role": "system", "content": EVALUATION_PROMT[self.task]},
            {"role": "user", "content": f"input: {x[0]}\n\noutput: {x[1]}"}
        ]
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=messages,
            temperature=0.6,
            top_p=0.9,
        )
        return response.choices[0].message.content
    
class Evaluation():
    def __init__(self, task, to_eval):
        self.task = task
        self.to_eval = to_eval

        self.dataset = load_dataset("csv", data_files={"test": f"human-evaluation/data/{self.task}.csv"})

        if self.to_eval == "gpt":
            self.gpt = GPT(self.task)
        elif self.to_eval == "llama":
            self.llama = LLaMA(self.task) 
        elif self.to_eval == "claude":
            self.claude = Claude(self.task) 

    def eval_llama(self, input, i):
        if os.path.exists(f"human-evaluation/llama/scores/{self.task}.csv"):
            df = pd.read_csv(f"human-evaluation/llama/scores/{self.task}.csv")
        else:
            df = pd.DataFrame(columns=["reasoning", "score"])

        if len(df) > i and df.iloc[i]["score"] != -1:
            pass
        else:
            output = self.llama.forward(input)

            output = ast.literal_eval(output)
            reasoning = output["reasoning"]
            score = output["score"]

            df.loc[i] = [reasoning, score]
            df.to_csv(f"human-evaluation/llama/scores/{self.task}.csv", index=False)


    def eval_gpt(self, input, i):
        if os.path.exists(f"human-evaluation/gpt/scores/{self.task}.csv"):
            df = pd.read_csv(f"human-evaluation/gpt/scores/{self.task}.csv")
        else:
            df = pd.DataFrame(columns=["reasoning", "score"])

        if len(df) > i and df.iloc[i]["score"] != -1:
            pass
        else:
            output = self.gpt.forward(input)

            output = ast.literal_eval(output)
            reasoning = output["reasoning"]
            score = output["score"]

            df.loc[i] = [reasoning, score]
            df.to_csv(f"human-evaluation/gpt/scores/{self.task}.csv", index=False)
    
    def eval_claude(self, input, i):
        if os.path.exists(f"human-evaluation/claude/scores/{self.task}.csv"):
            df = pd.read_csv(f"human-evaluation/claude/scores/{self.task}.csv")
        else:
            df = pd.DataFrame(columns=["reasoning", "score"])

        if len(df) > i and df.iloc[i]["score"] != -1:
            pass
        else:
            output = self.claude.forward(input)
            matches = re.findall(r"### Score: (\d{1,3})/100", output)
            score = matches[0]

            reasoning = output

            df.loc[i] = [reasoning, score]
            df.to_csv(f"human-evaluation/claude/scores/{self.task}.csv", index=False)

    def eval(self):
        k = 0
        for i in tqdm(range(50)):
            for j in range(0, 4):
                if self.to_eval == "gpt":
                    self.eval_gpt([self.dataset["test"][i]["Input"], self.dataset["test"][i][f"M{j + 1}"]], k)
            
                if self.to_eval == "llama":
                    self.eval_llama([self.dataset["test"][i]["Input"], self.dataset["test"][i][f"M{j + 1}"]], k)
                
                if self.to_eval == "claude":
                    self.eval_claude([self.dataset["test"][i]["Input"], self.dataset["test"][i][f"M{j + 1}"]], k)
                
                k += 1

    def generate_ranking(self):
        df = pd.DataFrame(columns=["Input", "M1", "M2", "M3", "M4"])
        k = 0
        if self.to_eval == "llama":
            df_scores = pd.read_csv(f"human-evaluation/llama/scores/{self.task}.csv")
        elif self.to_eval == "gpt":
            df_scores = pd.read_csv(f"human-evaluation/gpt/scores/{self.task}.csv")
        elif self.to_eval == "claude":
            df_scores = pd.read_csv(f"human-evaluation/claude/scores/{self.task}.csv")

        for i in range(0, len(df_scores), 4):
            scores = []
            scores.append(df_scores.iloc[i]["score"])
            scores.append(df_scores.iloc[i + 1]["score"])
            scores.append(df_scores.iloc[i + 2]["score"])
            scores.append(df_scores.iloc[i + 3]["score"])
            
            scores_array = np.array(scores)

            row = rankdata(-scores_array, method='min')    
            row = [self.dataset["test"][k]["Input"]] + list(row)

            df.loc[k] = row
            k += 1                
            
        if self.to_eval == "llama":
            df.to_csv(f"human-evaluation/llama/ranks/{self.task}.csv", index=False)
        elif self.to_eval == "gpt":
            df.to_csv(f"human-evaluation/gpt/ranks/{self.task}.csv", index=False)
        elif self.to_eval == "claude":
            df.to_csv(f"human-evaluation/claude/ranks/{self.task}.csv", index=False)

        

ev = Evaluation("translation-ur-en", "claude")
ev.eval()
ev.generate_ranking()