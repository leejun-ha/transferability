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

dna_vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4}

def tokenize_dna(seq):
    return [dna_vocab[base] for base in seq if base in dna_vocab]

def pad_sequence(seq, max_length):
    return seq[:max_length] + [4] * (max_length - len(seq))  # 4 as padding token

if not 'albert' in args.model:
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
elif 'albert' in args.model:
    tokenizer = transformers.AlbertTokenizerFast.from_pretrained(args.model)

input_lengths = [64, 128, 256, 384, 512]

def process_data(input_length):
    data = []
    label = []
    attention_mask = []
    
    with open(os.path.join(args.data_dir, f'{args.task}.txt')) as files:
        text = files.readlines()
        for i in tqdm(range(0, len(text), 3), desc=f"Processing for length {input_length}"):
            if i + 1 < len(text) and i + 2 < len(text):
                seq = text[i+1].strip().upper()
                label_value = int(text[i+2])
                
                tokens = tokenize_dna(seq)
                seq_chunks = [tokens[j:j+input_length] for j in range(0, len(tokens), input_length)]
                
                for chunk in seq_chunks:
                    padded_chunk = pad_sequence(chunk, input_length)
                    data.append(padded_chunk)
                    attention_mask.append([1] * len(chunk) + [0] * (input_length - len(chunk)))
                    label.append(label_value)

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

# Prepare log file
log_file = os.path.join(data_path, f'{args.task}_{model_replace}_preprocessing_log.txt')
with open(log_file, 'w') as log:
    log.write(f"Preprocessing Log for {args.task} using {args.model}\n")
    log.write(f"DNA Vocabulary: {dna_vocab}\n\n")

    for length in input_lengths:
        data, attention_mask, label = process_data(length)
        
        torch.save(data, os.path.join(data_path, f'{args.task}_{model_replace}_data_{length}.pkl'))
        torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{model_replace}_attention_mask_{length}.pkl'))
        torch.save(label, os.path.join(data_path, f'{args.task}_{model_replace}_label_{length}.pkl'))
        
        log.write(f"Input Length: {length}\n")
        log.write(f"Data shape: {data.shape}\n")
        log.write(f"Label shape: {label.shape}\n")
        log.write(f"Attention mask shape: {attention_mask.shape}\n")
        log.write(f"Sample data: {data[0][:10].tolist()}\n")
        log.write(f"Sample attention mask: {attention_mask[0][:10].tolist()}\n")
        log.write(f"Sample label: {label[0].item()}\n\n")

print("Processing completed for all input lengths.")
print(f"Preprocessing log saved to {log_file}")