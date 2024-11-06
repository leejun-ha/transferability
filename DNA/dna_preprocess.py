import torch
import numpy as np
import transformers
import random
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--data_dir', type = str, default = './Hilbert-CNN/data')
parser.add_argument('--save_dir', type = str, default = './data')
parser.add_argument('--model', type = str)
parser.add_argument('--seed', type = int, default = 100)
args = parser.parse_args()
#args = vars(args)
random.seed(args.seed)
data_path = os.path.join(args.save_dir, args.task)

dna_vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4}

def tokenize_dna(seq):
    return [dna_vocab[base] for base in seq if base in dna_vocab]

def pad_sequence(seq, max_length):
    return seq[:max_length] + [4] * (max_length - len(seq))  # 4 as padding token

if not os.path.exists(data_path):
    os.makedirs(data_path)

# if not 'albert' in args.model:
#     tokenizer = transformers.BertTokenizerFast.from_pretrained(args.model)
if not 'albert' in args.model:
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
elif 'albert' in args.model:
    tokenizer = transformers.AlbertTokenizerFast,from_pretrained(args.model)

tokenizer.model_max_length = 502

data = []
label = []
attention_mask = []
with open(os.path.join(args.data_dir, f'{args.task}.txt')) as files:
    text = files.readlines()
    max_length = 502
    for i in tqdm(range(len(text))):
        if i % 3 == 1:
            seq = text[i].strip().upper()  # remove newline and convert to uppercase
            tokens = tokenize_dna(seq)
            padded_tokens = pad_sequence(tokens, max_length)
            data.append(padded_tokens)
            attention_mask.append([1] * len(tokens) + [0] * (max_length - len(tokens)))
        elif i % 3 == 2:
            label.append(int(text[i]))

all_feature = list(zip(data, attention_mask, label))
random.shuffle(all_feature)
data, attention_mask, label = list(zip(*all_feature))
data = torch.LongTensor(data)
label = torch.LongTensor(label)
attention_mask = torch.Tensor(attention_mask)
print(data.shape)
# remove "/" in model name
model_replace = args.model
model_replace = model_replace.replace("/", "_")
print(model_replace)
torch.save(data, os.path.join(data_path, f'{args.task}_{model_replace}_data.pkl'))
torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{model_replace}_attention_mask.pkl'))
torch.save(label, os.path.join(data_path, f'{args.task}_{model_replace}_label.pkl'))

