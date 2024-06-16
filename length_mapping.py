LENGTH_MAPPING_MT5 = {
    "translation-en-ur": {"input": 128, "output": 128},
    "translation-ur-en": {"input": 128, "output": 128},
    "transliteration": {"input": 128, "output": 128},
    "summarization": {"input": 1024, "output": 256},
    "paraphrase": {"input": 128, "output": 128},
    "question-answering": {"input": 512, "output": 100},
}

LENGTH_MAPPING_LLAMA = {
    "translation-en-ur": 512,
    "translation-ur-en": 512,
    "transliteration": 512,
    "summarization": 1024,
    "paraphrase": 256,
    "question-answering": 128,
}