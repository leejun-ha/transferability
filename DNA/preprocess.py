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
    for i in tqdm(range(len(text))):
        if i%3 == 0:
            continue
        elif i%3 == 1:
            seq = " "
            seq = seq.join(text[i].replace('\n', ''))
            input_ids = tokenizer.encode_plus(seq, padding='max_length')
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
print(data.shape)
# remove "/" in model name
model_replace = args.model
model_replace = model_replace.replace("/", "_")
print(model_replace)
torch.save(data, os.path.join(data_path, f'{args.task}_{model_replace}_data.pkl'))
torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{model_replace}_attention_mask.pkl'))
torch.save(label, os.path.join(data_path, f'{args.task}_{model_replace}_label.pkl'))

