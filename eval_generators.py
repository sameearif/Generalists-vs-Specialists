from openai import OpenAI
from together import Together
import torch
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel   
from utils import *
from length_mapping import *
from datasets import load_dataset
import evaluate
import re
import ast
import os
import pandas as pd
from tqdm import tqdm

class GPT():
    def __init__(self, task):
        self.client = OpenAI(
                organization='org-OWe4x5PWbYh2yljPFA9odbCD',
            )
        self.task = task

    def forward(self, x, k, cot=False):
        if self.task == "translation-en-ur" or self.task == "translation-ur-en":
            messages = create_messages_translation(self.task, k, cot)
        elif self.task == "transliteration":
            messages = create_messages_transliteration(k, cot)
        elif self.task == "summarization":
            messages = create_messages_summarization(k, cot)
        elif self.task == "paraphrase":
            messages = create_messages_paraphrase(k, cot)
        elif self.task == "question-answering":
            messages = create_messages_question_answering(k, cot)
            x = f"Context:\n{x[0]}\n\nQuestion:\n{x[1]}"
        elif self.task == "ai-assistant":
            messages = [{"role": "system", "content": "Always answer in Pakistani Urdu language."}]
            x += "\n\nAnswer in Pakistani Urdu only."
        x = str(x)
        
        messages.append({"role": "user", "content": x})
        if self.task == "ai-assistant":
            completion = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                # response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        else:
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

    def forward(self, x, k=0, cot=False):
        if self.task == "translation-en-ur" or self.task == "translation-ur-en":
            messages = create_messages_translation(self.task, k, cot)
        elif self.task == "transliteration":
            messages = create_messages_transliteration(k, cot)
        elif self.task == "summarization":
            messages = create_messages_summarization(k, cot)
        elif self.task == "paraphrase":
            messages = create_messages_paraphrase(k, cot)
        elif self.task == "question-answering":
            messages = create_messages_question_answering(k, cot)
            x = f"Context:\n{x[0]}\n\nQuestion:\n{x[1]}"
        elif self.task == "ai-assistant":
            messages = [{"role": "system", "content": "Always answer in Pakistani Urdu language."}]
            x += "\n\nAnswer in Pakistani Urdu only."
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
        if self.task == "question-answering":
            input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{LLAMA_FT_PROMPTS[self.task]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{x[0]}\n\nQuestion:\n{x[1]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            return input
        elif self.task == "ai-assistant":
            input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            return input
        input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{LLAMA_FT_PROMPTS[self.task]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return input

    def forward(self, x):
        x = self.format_input(x)
        input_ids = self.tokenizer(x, return_tensors="pt").to(self.device)
        terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=LENGTH_MAPPING_LLAMA[self.task],
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators
        )
        output = self.tokenizer.decode(outputs[0])
        return output.split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "")

class MT5():
    def __init__(self, task, device):
        self.device = device
        self.task = task
        if self.task == "translation-en-ur" or self.task == "translation-ur-en":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"ULRs/mt5-large-{self.task}").to(self.device)
            self.tokenizer = MT5Tokenizer.from_pretrained(f"ULRs/mt5-large-{self.task}")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"ULRs/mt5-large-{self.task}-ur").to(self.device)
            self.tokenizer = MT5Tokenizer.from_pretrained(f"ULRs/mt5-large-{self.task}-ur")

    def forward(self, x):
        if self.task == "translation-en-ur" or self.task == "translation-ur-en":
            x = f"Translate: {x}"
        elif self.task == "transliteration":
            x = f"Transliterate: {x}"
        elif self.task == "summarization":
            x = f"Summarize: {x}"
        elif self.task == "paraphrase":
            x = f"Paraphrase: {x}"
        elif self.task == "question-answering":
            x = f"context: {x[0]} question: {x[1]}"
        elif self.task == "ai-assistant":
            x = f"Human: {x}"
        inputs = self.tokenizer.batch_encode_plus([x], truncation=True, padding="max_length", max_length=LENGTH_MAPPING_MT5[self.task]["input"], return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=LENGTH_MAPPING_MT5[self.task]["output"],
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output
    
class Metric():
    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")
        self.f1_score = evaluate.load("f1")
        self.squad_metric = evaluate.load("squad")
        self.rouge = evaluate.load("rouge")
    
    def calculate_bleu_score(self, predictions, references):
        return {"sacrebleu": self.bleu.compute(predictions=predictions, references=references)["score"]}
    
    def calculate_rouge_score(self, predictions, references):
        return self.rouge.compute(predictions=predictions, references=references, tokenizer=lambda x: x.split())
    
    def calculate_squad_metric(self, predictions, references):
        return self.squad_metric.compute(predictions=predictions, references=references)
        
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
        
        self.dataset = load_dataset("csv", data_files={"test": f"datasets/{self.task.replace('-en-ur', '').replace('-ur-en', '')}.csv"})

        if  self.to_eval == "gpt":
            self.gpt = GPT(self.task)
        if self.to_eval == "mt5":
            self.mt5 = MT5(self.task, device)
        if self.to_eval == "llama":
            self.llama = LLaMA(self.task)
        if self.to_eval == "llama-ft":
            self.llama_ft = LLaMAFT(self.task, device)
        
        self.metric = Metric()

    def extract_output(self, output, key):
        try:
            output = ast.literal_eval(output)[key]
        except:
            try:
                temp = "{" + re.search(rf"['\"]{key}['\"]\s*:\s*['\"].*?['\"]}}", output).group(0)
                output = ast.literal_eval(temp)[key]
            except:
                output = re.sub(r"[{}\[\]'\"':]", "", output.replace(key, "")).lstrip().rstrip()
        return output
    
    def eval_gpt(self, input, key, i):
        for k in [0, 3, 6]:
            if k not in self.k_shot:
                continue

            if os.path.exists(f"predictions/{self.task}/gpt-{k}-shot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/gpt-{k}-shot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])

            if len(df) > i and type(df.iloc[i]["prediction"]) == str:
                continue
            
            output = self.gpt.forward(input, k)
            if self.task != "ai-assistant":
                prediction = self.extract_output(output, key)
            else:
                prediction = output
            
            df.loc[i] = [prediction]
            df.to_csv(f"predictions/{self.task}/gpt-{k}-shot.csv", index=False, encoding="utf-8")
        
        if self.cot:
            if os.path.exists(f"predictions/{self.task}/gpt-cot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/gpt-cot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])
            
            if len(df) > i and type(df.iloc[i]["prediction"]) == str:
                pass
            else:
                output = self.gpt.forward(input, k, cot=True)
                prediction = self.extract_output(output, key)

                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/gpt-cot.csv", index=False, encoding="utf-8")
    
    def eval_llama(self, input, key, i):
        for k in [0, 3, 6]:
            if k not in self.k_shot:
                continue

            if os.path.exists(f"predictions/{self.task}/llama-{k}-shot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/llama-{k}-shot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])

            if len(df) > i and type(df.iloc[i]["prediction"]) == str:
                pass
            else:
                output = self.llama.forward(input, k)
                if self.task != "ai-assistant":
                    prediction = self.extract_output(output, key)
                else:
                    prediction = output
                
                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/llama-{k}-shot.csv", index=False, encoding="utf-8")
        
        if self.cot:
            if os.path.exists(f"predictions/{self.task}/llama-cot.csv"):
                df = pd.read_csv(f"predictions/{self.task}/llama-cot.csv")
            else:
                df = pd.DataFrame(columns=["prediction"])
            
            if len(df) > i and type(df.iloc[i]["prediction"]) == str:
                pass
            else:
                output = self.llama.forward(input, k, cot=True)
                prediction = self.extract_output(output, key)

                df.loc[i] = [prediction]
                df.to_csv(f"predictions/{self.task}/llama-cot.csv", index=False, encoding="utf-8")

    def eval_llama_ft(self, input, key, i):
        if os.path.exists(f"predictions/{self.task}/llama-ft.csv"):
            df = pd.read_csv(f"predictions/{self.task}/llama-ft.csv")
        else:
            df = pd.DataFrame(columns=["prediction"])

        if len(df) > i and type(df.iloc[i]["prediction"]) == str:
            pass
        else:
            output = self.llama_ft.forward(input)
            if self.task != "ai-assistant":
                prediction = self.extract_output(output, key)
            else:
                prediction = output

            df.loc[i] = [prediction]
            df.to_csv(f"predictions/{self.task}/llama-ft.csv", index=False, encoding="utf-8")


    def eval_mt5(self, input, i):
        if os.path.exists(f"predictions/{self.task}/mt5.csv"):
            df = pd.read_csv(f"predictions/{self.task}/mt5.csv")
        else:
            df = pd.DataFrame(columns=["prediction"])

        if len(df) > i:
            pass
        else:
            prediction = self.mt5.forward(input)
            if self.task == "ai-assistant":
                prediction = prediction.replace("Assistant: ", "")

            df.loc[i] = [prediction]
            df.to_csv(f"predictions/{self.task}/mt5.csv", index=False, encoding="utf-8")
        
    def eval(self):    
        for i in tqdm(range(len(self.dataset["test"]))):
            key = ""
            if self.task == "translation-en-ur":
                input = self.dataset["test"][i]["english"]
                key = "translation"
            elif self.task == "translation-ur-en":
                input = self.dataset["test"][i]["urdu"]
                key = "translation"
            elif self.task == "transliteration":
                input = self.dataset["test"][i]["urdu"]
                key = "transliteration"
            elif self.task == "summarization":
                input = self.dataset["test"][i]["text"]
                key = "summary"
            elif self.task == "paraphrase":
                input = self.dataset["test"][i]["text"]
                key = "paraphrase"
            elif self.task == "question-answering":
                input = [self.dataset["test"][i]["context"], self.dataset["test"][i]["question"]]
                key = "answer"
            elif self.task == "ai-assistant":
                input = self.dataset["test"][i]["input"]

            if self.to_eval == "gpt":
                self.eval_gpt(input, key, i)

            if self.to_eval == "llama":
                self.eval_llama(input, key, i)

            if self.to_eval == "llama-ft":
              self.eval_llama_ft(input, key, i)

            if self.to_eval == "mt5":
                self.eval_mt5(input, i)
    
    def calcuate_metric(self):
        if self.task == "question-answering":
            squad = {}
            true = self.dataset["test"]["answer"]
            y_true = []
            for i in true:
                y_true.append(ast.literal_eval(i))
            for model in ["gpt-0-shot", "gpt-3-shot", "gpt-6-shot", "gpt-cot", "llama-0-shot", "llama-3-shot", "llama-6-shot", "llama-cot", "llama-ft", "xlm-roberta"]:
                if os.path.exists(f"predictions/{self.task}/{model}.csv"):
                    df = pd.read_csv(f"predictions/{self.task}/{model}.csv")
                    y = df["prediction"].tolist()
                    predictions = []
                    references = []
                    for i, y_ in enumerate(y):
                        predictions.append({"id": str(i), "prediction_text": y_})
                        temp = []
                        for ref_ in y_true[i]:
                            temp.append({"text": ref_, "answer_start": 0})
                        references.append({"id": str(i), "answers": temp})
                    squad[model] = self.metric.calculate_squad_metric(predictions, references)
            return squad
        elif self.task == "summarization":
            rouge = {}
            references = self.dataset["test"]["summary"]
            for model in ["gpt-0-shot", "gpt-3-shot", "gpt-6-shot", "gpt-cot", "llama-0-shot", "llama-3-shot", "llama-6-shot", "llama-cot", "llama-ft", "xlm-roberta"]:
                if os.path.exists(f"predictions/{self.task}/{model}.csv"):
                    df = pd.read_csv(f"predictions/{self.task}/{model}.csv")
                    predictions = df["prediction"].tolist()
                    rouge[model] = self.metric.calculate_rouge_score(predictions, references)
            return rouge
        else:
            bleu = {}
            if self.task == "transliteration" or self.task == "translation-ur-en":
                key = "english"
            elif self.task == "paraphrase":
                key = "paraphrase"
            elif self.task == "translation-en-ur":
                key = "urdu"
            references = self.dataset["test"][key]
            for model in ["gpt-0-shot", "gpt-3-shot", "gpt-6-shot", "gpt-cot", "llama-0-shot", "llama-3-shot", "llama-6-shot", "llama-cot", "llama-ft", "xlm-roberta"]:
                if os.path.exists(f"predictions/{self.task}/{model}.csv"):
                    df = pd.read_csv(f"predictions/{self.task}/{model}.csv")
                    predictions = df["prediction"].tolist()
                    bleu[model] = self.metric.calculate_bleu_score(predictions, references)
            return bleu


ev = Evaluation("ai-assistant", [0], False, "mt5", "mps")
ev.eval()
# print(ev.calcuate_metric())
