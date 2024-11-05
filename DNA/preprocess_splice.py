import torch
import numpy as np
import transformers
import random
import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, default = 'splice')
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

if not 'albert' in args.model:
    tokenizer = transformers.BertTokenizerFast.from_pretrained(args.model)
else:
    tokenizer = transformers.AlbertTokenizerFast,from_pretrained(args.model)
tokenizer.model_max_length = 62

label_map = { 'EI': 0, 
              'IE': 1,
              'N': 2}

data = []
label = []
attention_mask = []
with open(os.path.join(args.data_dir, f'{args.task}.txt')) as files:
    text = files.readlines()
    for i in tqdm(range(len(text))):
        one_label, data_id, dna = text[i].split(",")
        dna = dna.lstrip().replace('\n', '')
        seq = " "
        seq = seq.join(dna)
        input_ids = tokenizer.encode_plus(seq, padding='max_length')
        data.append(input_ids['input_ids'])
        attention_mask.append(input_ids['attention_mask'])
        label.append(label_map[one_label])

all_feature = list(zip(data, attention_mask, label))
random.shuffle(all_feature)
data, attention_mask, label = list(zip(*all_feature))
data = torch.LongTensor(data)
attention_mask = torch.Tensor(attention_mask)
label = torch.LongTensor(label)
print(data.shape)
torch.save(data, os.path.join(data_path, f'{args.task}_{args.model}_data.pkl'))
torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{args.model}_attention_mask.pkl'))
torch.save(label, os.path.join(data_path, f'{args.task}_{args.model}_label.pkl'))

