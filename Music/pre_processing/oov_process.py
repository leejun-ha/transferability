import pretty_midi
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='maestro-v1')
parser.add_argument('--data_dir', type=str, default='../data/maestro-v1.0.0')
parser.add_argument('--save_dir', type=str, default='../data')
parser.add_argument('--model', type=str)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--token_len', type=int, default=128)
parser.add_argument('--json_file', type=str, default='maestro_v1_test_token_freq.json')
args = parser.parse_args()

import json

def load_and_process_frequency_data(json_file):
    with open(json_file, 'r') as f:
        frequency_data = json.load(f)
    
    sorted_tokens = sorted(frequency_data.items(), key=lambda x: x[1])
    total_freq = sum(freq for _, freq in sorted_tokens)
    cumulative_freq = 0
    tokens_to_remove = set()

    for token, freq in sorted_tokens:
        cumulative_freq += freq
        tokens_to_remove.add(int(token.split('_')[-1]))
        if cumulative_freq >= total_freq * 0.5:
            break

    return tokens_to_remove

def get_pitch(midi_data, token_len, tokens_to_remove):
    note_list = []
    instrument = midi_data.instruments
    notes = instrument[0].notes
    i = 0
    
    while (i+1)*token_len < len(notes):
        note = sorted(notes[i*token_len:(i+1)*token_len], key=lambda x:x.start)
        note = [n.pitch for n in note if n.pitch not in tokens_to_remove]
        if len(note) == token_len:  # Only keep sequences with the full token length
            note_list.append(note)
        i += 1
    return note_list

csv = pd.read_csv(f'../data/maestro-v1.0.0/{args.task}.0.0.csv')

composer = csv['canonical_composer']
split = csv['split']
midi_filename = csv['midi_filename']
composer2id = {}
idx = 0
for k in range(len(composer)):
    if composer[k] not in composer2id.keys():
        composer2id[composer[k]] = idx
        idx += 1

print("composer_num: ", idx)

# Load and process the frequency data
tokens_to_remove = load_and_process_frequency_data(args.json_file)

train_data = []
train_label = []
dev_data = []
dev_label = []
test_data = []
test_label = []

token_len = args.token_len

for i in tqdm(range(len(composer))):
    midi_data = pretty_midi.PrettyMIDI(f'../data/{args.task}.0.0/{midi_filename[i]}')
    pitch = get_pitch(midi_data, token_len, tokens_to_remove)
    label = composer2id[composer[i]]
    if split[i] == 'train':
        train_data.extend(pitch)
        train_label.extend([label]*len(pitch))
    elif split[i] == 'validation':
        dev_data.extend(pitch)
        dev_label.extend([label]*len(pitch))
    elif split[i] == 'test':
        test_data.extend(pitch)
        test_label.extend([label]*len(pitch))
    else:
        raise NotImplementedError

# Convert to tensors and save
train_t = (torch.Tensor(train_data)+128).long()
dev_t = (torch.Tensor(dev_data)+128).long()
test_t = (torch.Tensor(test_data)+128).long()

model_process_path = f'../data/pkl'


torch.save(train_t, model_process_path + f'/{token_len}_train_data_filtered.pkl')
torch.save(torch.LongTensor(train_label), model_process_path + f'/{token_len}_train_label_filtered.pkl')
torch.save(dev_t, model_process_path + f'/{token_len}_dev_data_filtered.pkl')
torch.save(torch.LongTensor(dev_label), model_process_path + f'/{token_len}_dev_label_filtered.pkl')
torch.save(test_t, model_process_path + f'/{token_len}_test_data_filtered.pkl')
torch.save(torch.LongTensor(test_label), model_process_path + f'/{token_len}_test_label_filtered.pkl')
torch.save(composer2id, model_process_path + f'/composer2id_map_filtered.pkl')