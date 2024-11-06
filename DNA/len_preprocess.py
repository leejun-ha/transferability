import torch
import numpy as np
import transformers
import random
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--data_dir', type=str, default='./Hilbert-CNN/data')
parser.add_argument('--save_dir', type=str, default='./data')
parser.add_argument('--model', type=str)
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

random.seed(args.seed)
data_path = os.path.join(args.save_dir, args.task)

if not os.path.exists(data_path):
    os.makedirs(data_path)

if 'albert' in args.model:
    tokenizer = transformers.AlbertTokenizerFast.from_pretrained(args.model)
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

input_lengths = [64, 128, 256, 384, 512]

def process_sequence(seq, max_length):
    if len(seq) > max_length:
        return seq[:max_length]
    return seq

def process_data(input_length):
    tokenizer.model_max_length = input_length
    data = []
    label = []
    attention_mask = []
    
    with open(os.path.join(args.data_dir, f'{args.task}.txt')) as files:
        text = files.readlines()
        for i in tqdm(range(len(text)), desc=f"Processing for length {input_length}"):
            if i % 3 == 0:
                continue
            elif i % 3 == 1:
                seq = " ".join(text[i].replace('\n', ''))
                seq = process_sequence(seq, input_length - 2)  # Account for [CLS] and [SEP] tokens
                input_ids = tokenizer.encode_plus(seq, padding='max_length', max_length=input_length, truncation=True)
                data.append(input_ids['input_ids'])
                attention_mask.append(input_ids['attention_mask'])
            else:
                label.append(int(text[i]))

    all_feature = list(zip(data, attention_mask, label))
    random.shuffle(all_feature)
    data, attention_mask, label = list(zip(*all_feature))
    data = torch.LongTensor(data)
    label = torch.LongTensor(label)
    attention_mask = torch.Tensor(attention_mask)
    print(f"Data shape for length {input_length}: {data.shape}")

    return data, attention_mask, label

model_replace = args.model.replace("/", "_")
print(f"Model: {model_replace}")

for length in input_lengths:
    data, attention_mask, label = process_data(length)
    
    torch.save(data, os.path.join(data_path, f'{args.task}_{model_replace}_data_{length}.pkl'))
    torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{model_replace}_attention_mask_{length}.pkl'))
    torch.save(label, os.path.join(data_path, f'{args.task}_{model_replace}_label_{length}.pkl'))

print("Processing completed for all input lengths.")