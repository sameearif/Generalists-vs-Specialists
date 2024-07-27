PROMPTS = {
"sentiment-analysis": """You are an Urdu sentiment classifier. The input text should be labeled according to its sentiment. The label list is:
['positive', 'negative']. Do not assign label "neutral" to the input text.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"abuse-detection": """You are an Urdu abuse detector. The input text should be labeled according to whether it is abusive or not. The label list is:
['abusive', 'not abusive']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"sarcasm-detection": """You are an Urdu sarcasm detector. The input text should be labeled according to whether it is sarcastic or not. The label list is:
['sarcastic', 'not sarcastic']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"fake-news-detection": """You are an Urdu fake news detector. The input text should be labeled according to whether it is fake news or not. The label list is:
['fake news', 'not fake news']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"topic-classification": """You are an Urdu topic classifier. The input text should be assign a label from the label list:
['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'world', 'other']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"ner-tagging": """You are an Urdu named entity recognizer. The word highlighted with <hl> tag should be assigned a NER tag from the label list:
['time', 'person', 'organization', 'number', 'location', 'designation', 'date', 'other']
Assign the label to the highlighted word only.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"pos-tagging": """You are an Urdu part-of-speech tagger. The word highlighted with <hl> tag should be assigned a PoS tag from the label list:
['noun', 'punctuation mark', 'adposition', 'number', 'symbol', 'subordinating conjunction', 'adjective', 'particle', 'determiner', 'coordinating conjunction', 'proper noun', 'pronoun', 'other', 'adverb', 'interjection', 'verb', 'auxiliary verb']
Assign the label to the highlighted word only.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"translation-en-ur": """You are a machine translator. Your task is to translate the given English text to Pakistani Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"translation": ...}""",

"translation-ur-en": """You are a machine translator. Your task is to translate the given Pakistani Urdu text to English.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"translation": ...}""",

"transliteration": """You are a machine transliterator. Your task is to transliterate the given Pakistani Urdu text to Roman Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"transliteration": ...}""",

"summarization": """You are a text summarizer. Your task is to summarize the given Pakistani Urdu text in 1 to 2 sentences. The summary should be in Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"summary": ...}""",

"paraphrase": """You are a text paraphraser. Your task is to paraphrase the given Pakistani Urdu text. The paraphrased text should be in Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"paraphrase": ...}""",

"question-answering": """You are a question answering model. Your task is to answer the given question from the given context ONLY. Don't return null, always answer the question.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"answer": ...}"""
}

LLAMA_FT_PROMPTS = {
"sentiment-analysis": """You are an Urdu sentiment classifier. The input text should be labeled according to its sentiment. The label list is:
['positive', 'negative']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"abuse-detection": """You are an Urdu abuse detector. The input text should be labeled according to whether it is abusive or not. The label list is:
['abusive', 'not abusive']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"sarcasm-detection": """You are an Urdu sarcasm detector. The input text should be labeled according to whether it is sarcastic or not. The label list is:
['sarcastic', 'not sarcastic']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"fake-news-detection": """You are an Urdu fake news detector. The input text should be labeled according to whether it is fake news or not. The label list is:
['fake news', 'not fake news']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"topic-classification": """You are an Urdu topic classifier. The input text should be assign a label from the label list:
['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'world', 'other']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"ner-tagging": """You are an Urdu named entity recognizer. The word highlighted with <hl> tag should be assigned a NER tag from the label list:
['time', 'person', 'organization', 'number', 'location', 'designation', 'date', 'other']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"pos-tagging": """You are an Urdu part-of-speech tagger. The word highlighted with <hl> tag should be assigned a PoS tag from the label list:
['noun', 'punctuation mark', 'adposition', 'number', 'symbol', 'subordinating conjunction', 'adjective', 'particle', 'determiner', 'coordinating conjunction', 'proper noun', 'pronoun', 'other', 'adverb', 'interjection', 'verb', 'auxiliary verb']
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"label": ...}""",

"translation-en-ur": """You are a machine translator. Your task is to translate the given English text to Pakistani Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"translation": ...}""",

"translation-ur-en": """You are a machine translator. Your task is to translate the given Pakistani Urdu text to English.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"translation": ...}""",

"transliteration": """You are a machine transliterator. Your task is to transliterate the given Pakistani Urdu text to Roman Urdu.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"transliteration": ...}""",

"summarization": """You are a text summarizer. Your task is to summarize the given Pakistani Urdu text in 1 to 2 sentences.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"summary": ...}""",

"paraphrase": """You are a text paraphraser. Your task is to paraphrase the given Pakistani Urdu text.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"paraphrase": ...}""",

"question-answering": """You are a question answering model. Your task is to answer the given question from the given context ONLY.
ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"answer": ...}"""
}

COT_PROMPTS = {
"sentiment-analysis": """You are an Urdu sentiment classifier. The input text should be labeled according to its sentiment. The label list is:
['positive', 'negative']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"abuse-detection": """You are an Urdu abuse detector. The input text should be labeled according to whether it is abusive or not. The label list is:
['abusive', 'not abusive']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"sarcasm-detection": """You are an Urdu sarcasm detector. The input text should be labeled according to whether it is sarcastic or not. The label list is:
['sarcastic', 'not sarcastic']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"fake-news-detection": """You are an Urdu fake news detector. The input text should be labeled according to whether it is fake news or not. The label list is:
['fake news', 'not fake news']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"topic-classification": """You are an Urdu topic classifier. The input text should be assign a label from the label list:
['business', 'entertainment', 'health', 'politics', 'science', 'sports', 'world', 'other']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"pos-tagging": """You are an Urdu part-of-speech tagger. The word wrapped in <hl> tag should be assigned a PoS tag from the label list:
['noun', 'punctuation mark', 'adposition', 'number', 'symbol', 'subordinating conjunction', 'adjective', 'particle', 'determiner', 'coordinating conjunction', 'proper noun', 'pronoun', 'other', 'adverb', 'interjection', 'verb', 'auxiliary verb']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}""",

"ner-tagging": """You are an Urdu named entity recognizer. The word wrapped in <hl> tag should be assigned a NER tag from the label list:
['time', 'person', 'organization', 'number', 'location', 'designation', 'date', 'other']
Use chain of thought (cot) to reason your answer. ALWAYS RETURN JSON OBJECT IN FOLLOWING FORMAT ONLY:
{"cot": ..., "label": ...}"""
}

EVALUATION_PROMT = {
    "paraphrase": """You are a Pakistani Urdu language expert tasked with evaluating the quality of paraphrased text produced by the paraphrasing model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Retains the original meaning and key ideas
    2. No grammatical errors
    3. Use of different words and phrases than the original text
    Think step by step and use reasoning. ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}""",

    "summarization": """You are a Pakistani Urdu language expert tasked with evaluating the quality of summary produced by the summarization model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Includes main points and key information from the original text
    2. No grammatical errors
    3. Conveys information in a brief manner
    4. Gives correct information based on the original text
    Think step by step and use reasoning (reasoning should not be too big). ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}""",

    "transliteration": """You are a Pakistani Urdu language expert tasked with evaluating the quality of transliterated text produced by the transliteration model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Correctness of words (keeping  in mind that different words can have different spelling)
    2. Proper capitalisation of words
    Think step by step and use reasoning. ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}""",

    "translation-en-ur": """You are a Pakistani Urdu language expert tasked with evaluating the quality of translated text produced by the translation model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Conveys the meaning of the original text without omissions or additions
    2. No grammatical errors
    3. Retains the style and tone of the original text
    Think step by step and use reasoning. ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}""",

    "translation-ur-en": """You are a Pakistani Urdu language expert tasked with evaluating the quality of translated text produced by the translation model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Conveys the meaning of the original text without omissions or additions
    2. No grammatical errors
    3. Retains the style and tone of the original text
    Think step by step and use reasoning. ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}""",

    "ai-assistant": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 100. ALWAYS RETURN JSON OBJECT IN THE FOLLOWING FORMAT ONLY:
    {"reasoning": ..., "score": ...}"""
}

CLAUDE_EVALUATION_PROMT = {
    "paraphrase": """You are a Pakistani Urdu language expert tasked with evaluating the quality of paraphrased text produced by the paraphrasing model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Retains the original meaning and key ideas
    2. No grammatical errors
    3. Use of different words and phrases than the original text
    Think step by step and use reasoning. ALWAYS RETURN YOUR ANSWER IN THE FOLLOWING FORMAT ONLY:
    ### Reasoning:
    [Add your reasoning here]
    ### Score: X/100""",

    "summarization": """You are a Pakistani Urdu language expert tasked with evaluating the quality of summary produced by the summarization model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Includes main points and key information from the original text
    2. No grammatical errors
    3. Conveys information in a brief manner
    4. Gives correct information based on the original text
    Think step by step and use reasoning (reasoning should not be too big). ALWAYS RETURN YOUR ANSWER IN THE FOLLOWING FORMAT ONLY:
    ### Reasoning:
    [Add your reasoning here]
    ### Score: X/100""",

    "transliteration": """You are a Pakistani Urdu language expert tasked with evaluating the quality of transliterated text produced by the transliteration model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Correctness of words (keeping  in mind that different words can have different spelling)
    2. Proper capitalisation of words
    Think step by step and use reasoning. ALWAYS RETURN YOUR ANSWER IN THE FOLLOWING FORMAT ONLY:
    ### Reasoning:
    [Add your reasoning here]
    ### Score: X/100""",

    "translation-en-ur": """You are a Pakistani Urdu language expert tasked with evaluating the quality of translated text produced by the translation model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Conveys the meaning of the original text without omissions or additions
    2. No grammatical errors
    3. Retains the style and tone of the original text
    Think step by step and use reasoning. ALWAYS RETURN YOUR ANSWER IN THE FOLLOWING FORMAT ONLY:
    ### Reasoning:
    [Add your reasoning here]
    ### Score: X/100""",

    "translation-ur-en": """You are a Pakistani Urdu language expert tasked with evaluating the quality of translated text produced by the translation model. Score the given output with respect to the given input on a continuous score from 0 to 100 based on the following criteria:
    1. Conveys the meaning of the original text without omissions or additions
    2. No grammatical errors
    3. Retains the style and tone of the original text
    Think step by step and use reasoning. ALWAYS RETURN YOUR ANSWER IN THE FOLLOWING FORMAT ONLY:
    ### Reasoning:
    [Add your reasoning here]
    ### Score: X/100""",

    "ai-assistant": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 100 by strictly following this format:
    ### Reasoning:
    [Give reasoning in English]
    ### Score: X/100"""
}

