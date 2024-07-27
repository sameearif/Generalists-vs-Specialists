from openai import OpenAI
from together import Together
import torch
from transformers import AutoModelForSequenceClassification, XLMRobertaForTokenClassification, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel    
from utils import *
from length_mapping import *
from datasets import load_dataset
import evaluate
import re
import ast
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

class GPT():
    def __init__(self, task):
        self.client = OpenAI(
                organization='org-OWe4x5PWbYh2yljPFA9odbCD',
            )
        self.task = task

    def forward(self, x, k, cot=False):
        if cot:
            messages = create_messages_cot_classifier(self.task)
        else:
            messages = create_messages_classifier(self.task, k)
        x = str(x)
        messages.append({"role": "user", "content": x})
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content
    
class LLaMA():
    def __init__(self, task):
        self.client = Together()
        self.task = task
    
    def forward(self, x, k, cot=False):
        if cot:
            messages = create_messages_cot_classifier(self.task)
        else:
            messages = create_messages_classifier(self.task, k)
        x = str(x)
        messages.append({"role": "user", "content": x})
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",
                messages=messages,
                temperature=0.6,
                top_p=0.9,
            )
            return response.choices[0].message.content
        except:
            return ""

class LLaMAFT():
    def __init__(self, task, device):
        self.task = task
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=self.bnb_config)
        self.model = PeftModel.from_pretrained(self.model, f"ULRs/llama-3-8b-{task}-ur").to(self.device)

    def format_input(self, x):
        input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{LLAMA_FT_PROMPTS[self.task]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return input

    def forward(self, x):
        x = self.format_input(x)
        input_ids = self.tokenizer(x, return_tensors="pt").to(self.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators
        )
        output = self.tokenizer.decode(outputs[0])
        return output.split("assistant<|end_header_id|>\n\n")[1].replace("<|eot_id|>", "")

class XLMR():
    def __init__(self, task, device):
        self.device = device
        self.task = task
        if task == "ner-tagging" or task == "pos-tagging":
            self.model = XLMRobertaForTokenClassification.from_pretrained(f"ULRs/xlm-roberta-large-{self.task}-ur").to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(f"ULRs/xlm-roberta-large-{self.task}-ur").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"ULRs/xlm-roberta-large-{self.task}-ur")
    
    def forward(self, x):
        if self.task == "ner-tagging" or self.task == "pos-tagging":
            inputs = self.tokenizer(x, add_special_tokens=False, is_split_into_words=True, return_tensors="pt")
        else:
            inputs = self.tokenizer(x, max_length=512, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits
        output = np.argmax(logits.cpu().detach().numpy(), axis=-1)[0]

        if (self.task == "ner-tagging" or self.task == "pos-tagging") and len(output) != len(x):
            new_output = []
            prev_token = None
            for i, token_id in enumerate(inputs.input_ids[0]):
                token = self.tokenizer.decode(token_id)
                if token != x[0]:
                    if token == "االله" and x[0] == "اﷲ":
                        x = x[1:]
                        new_output.append(output[i])
                    elif prev_token is None:
                        prev_token = token
                    elif prev_token + token == x[0]:
                        x = x[1:]
                        new_output.append(output[i])
                        prev_token = None
                    elif prev_token + token == "صلىاللهعليهوسلم":
                        x = x[1:]
                        new_output.append(output[i])
                        prev_token = None
                    elif prev_token + token == "اسلامصلىاللهعليهوسلم":
                        x = x[1:]
                        new_output.append(output[i])
                        prev_token = None
                    elif prev_token + token == "عبداالله":
                        x = x[1:]
                        new_output.append(output[i])
                        prev_token = None
                    elif prev_token + token == "لايٴن":
                        x = x[1:]
                        new_output.append(output[i])
                        prev_token = None
                    else:
                        prev_token += token
                else:
                    x = x[1:]
                    new_output.append(output[i])
            return new_output
        else:
            return output
        
class Metric():
    def __init__(self):
        self.accuracy = evaluate.load("accuracy")
        self.f1_score = evaluate.load("f1")
    
    def calculate_accuracy(self, predictions, references):
        return self.accuracy.compute(predictions=predictions, references=references)
        
    def calculate_f1_score(self, predictions, references):
        return self.f1_score.compute(predictions=predictions, references=references, average="macro")

class Evaluation():
    def __init__(self, task, k_shot, cot, to_eval, device):
        self.task = task
        if not os.path.exists(f"predictions/{self.task}"):
            os.mkdir(f"predictions/{self.task}")                  
        self.k_shot = k_shot
        self.cot = cot
        self.to_eval = to_eval

        self.dataset = load_dataset("csv", data_files={"test": f"datasets/{self.task}.csv"})

        if self.to_eval == "gpt":
            self.gpt = GPT(self.task)
        elif self.to_eval == "llama":
            self.llama = LLaMA(self.task) 
        elif self.to_eval == "llama-ft":
            self.llama_ft = LLaMAFT(self.task, device)
        elif self.to_eval == "xlm-roberta":
            self.xlmr = XLMR(self.task, device)

        self.metric = Metric()

    def extract_label(self, output):
        try:
            output = ast.literal_eval(output)["label"]
        except:
            try:
                temp = "{" + re.search(rf"['\"]label['\"]\s*:\s*['\"].*?['\"]}}", output).group(0)
                output = ast.literal_eval(temp)["label"]
            except:
                if self.task == "abuse-detection":
                    output = check_abusive_status(output)
                elif self.task == "sarcasm-detection":
                    output = check_sarcastic_status(output)
                elif self.task == "fake-news":
                    output = check_fake_news_status(output)
                else:
                    labels = label_mapping[self.task]
                    pred_label = None
                    for label in labels:
                        if label in output.lower():
                            if pred_label is None:
                                output = str({'label': label})
                                pred_label = label
                            else:
                                output = ""
                                break
                try:
                    output = ast.literal_eval(output)["label"]
                except:
                    output = ""

        if output != "":
            try:
                prediction = label_mapping[self.task][output.lower()]
            except:
                prediction = -1
        else:
            prediction = -1

        return prediction
    
    def eval_gpt(self, input, i):
        for k in [0, 3, 6]:
            if k not in self.k_shot:
                continue

            if os.path.exists(f"predictions/{self.task}/gpt-{k}-shot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/gpt-{k}-shot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])
            
            if len(df) > i and df.iloc[i]["prediction"] != -1:
                continue
                
            if self.task == "ner-tagging" or self.task == "pos-tagging":
                output = self.gpt.forward(highlight_word(input, self.dataset["test"][i]["index"]), k)
            else:
                output = self.gpt.forward(input, k)
            
            prediction = self.extract_label(output)

            df.loc[i] = [prediction]
            df.to_csv(f"predictions/{self.task}/gpt-{k}-shot.csv", index=False)
        
        if self.cot:
            if os.path.exists(f"predictions/{self.task}/gpt-cot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/gpt-cot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])

            if len(df) > i and df.iloc[i]["prediction"] != -1:
                pass
            else:
                if self.task == "ner-tagging" or self.task == "pos-tagging":
                    output = self.gpt.forward(highlight_word(input, self.dataset["test"][i]["index"]), k, cot=True)
                else:
                    output = self.gpt.forward(input, k, cot=True)

                prediction = self.extract_label(output)

                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/gpt-cot.csv", index=False)

    def eval_llama(self, input, i):
        for k in [0, 3, 6]:
            if k not in self.k_shot:
                continue
            if os.path.exists(f"predictions/{self.task}/llama-{k}-shot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/llama-{k}-shot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])
            
            if len(df) > i and df.iloc[i]["prediction"] != -1:
                pass
            else:
                if self.task == "ner-tagging" or self.task == "pos-tagging":
                    output = self.llama.forward(highlight_word(input, self.dataset["test"][i]["index"]), k)
                else:
                    output = self.llama.forward(input, k)

                prediction = self.extract_label(output)

                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/llama-{k}-shot.csv", index=False)
        
        if self.cot:
            if os.path.exists(f"predictions/{self.task}/llama-cot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/llama-cot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])

            if len(df) > i and df.iloc[i]["prediction"] != -1:
                pass
            else:
                if self.task == "ner-tagging" or self.task == "pos-tagging":
                    output = self.llama.forward(highlight_word(input, self.dataset["test"][i]["index"]), k, cot=True)
                else:
                    output = self.llama.forward(input, k, cot=True)

                prediction = self.extract_label(output)

                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/llama-cot.csv", index=False)
        
    def eval_llama_ft(self, input, i):
        if os.path.exists(f"predictions/{self.task}/llama-ft.csv"):
            df = pd.read_csv(f"predictions/{self.task}/llama-ft.csv")
        else:
            df = pd.DataFrame(columns=["prediction"])
        
        if len(df) > i and df.iloc[i]["prediction"] != -1:
            pass
        else:
            if self.task == "ner-tagging" or self.task == "pos-tagging":
                output = self.llama_ft.forward(highlight_word(input, self.dataset["test"][i]["index"]))
            else:
                output = self.llama_ft.forward(input)
            
            prediction = self.extract_label(output)

            df.loc[i] = [prediction]
            df.to_csv(f"predictions/{self.task}/llama-ft.csv", index=False)
        

    def eval_xlmr(self, input, i):
        if os.path.exists(f"predictions/{self.task}/xlm-roberta.csv"):
            df = pd.read_csv(f"predictions/{self.task}/xlm-roberta.csv")
        else:
            df = pd.DataFrame(columns=["prediction"])

        if len(df) > i and type(df.iloc[i]["prediction"]) == np.int64 and df.iloc[i]["prediction"] != -1:
            pass
        else:
            output = self.xlmr.forward(input)
            prediction = output
            df.loc[i] = prediction[self.dataset["test"][i]]
            df.to_csv(f"predictions/{self.task}/xlm-roberta.csv", index=False)
    
    def eval(self):    
        for i in tqdm(range(len(self.dataset["test"]))):
            if self.task == "ner-tagging" or self.task == "pos-tagging":
                input = string_to_list(self.dataset["test"][i]["input"])
            else:
                input = self.dataset["test"][i]["input"]

            if self.to_eval == "gpt":
                self.eval_gpt(input, i)
            
            if self.to_eval == "llama":
                self.eval_llama(input, i)
            
            if self.to_eval == "llama-ft":
                self.eval_llama_ft(input, i)

            if self.to_eval == "xlm-roberta":
                self.eval_xlmr(input, i)
        
    def calcuate_metric(self):
        f1 = {}
        accuracy = {}
        references = self.dataset["test"]["label"]
        for model in ["gpt-0-shot", "gpt-3-shot", "gpt-6-shot", "gpt-cot", "llama-0-shot", "llama-3-shot", "llama-6-shot", "llama-cot", "llama-ft", "xlm-roberta"]:
             if os.path.exists(f"predictions/{self.task}/{model}.csv"):
                df = pd.read_csv(f"predictions/{self.task}/{model}.csv")
                predictions = df["prediction"].tolist()
                f1[model] = self.metric.calculate_f1_score(predictions, references)
                accuracy[model] = self.metric.calculate_accuracy(predictions, references)

        return f1, accuracy
    

ev = Evaluation("ner-tagging", [], True, "llama", "mps")
ev.eval()
f1, accuracy = ev.calcuate_metric()
print(f1)
print("=========================")
# print(accuracy)
