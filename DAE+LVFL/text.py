import torch
from transformers import BertTokenizer

def text_prompt(data):
    tokenizer = BertTokenizer.from_pretrained('./bert/')


    text_input = tokenizer(data, return_tensors='pt', padding='max_length', truncation=True,max_length=100)
    text_input = {key: value.cuda() for key, value in text_input.items()}

    return text_input





