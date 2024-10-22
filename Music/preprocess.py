import pretty_midi
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, default= 'maestro-v1')
parser.add_argument('--data_dir', type = str, default = './data/maestro-v1.0.0')
parser.add_argument('--save_dir', type = str, default = './data')
parser.add_argument('--model', type = str)
parser.add_argument('--seed', type = int, default = 100)
parser.add_argument('--token_len', type = int, default = 128)
args = parser.parse_args()

csv = pd.read_csv(f'./{args.task}.0.0.csv')
                  
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
        
def get_pitch(midi_data, token_len):
    note_list = []
    instrument = midi_data.instruments
    notes = instrument[0].notes
    i = 0
    
    while (i+1)*token_len < len(notes):
        note = sorted(notes[i*token_len:(i+1)*token_len], key=lambda x:x.start)
        note = [ n.pitch for n in note ]
        note_list.append(note)
        i += 1
    return note_list

train_data = []
train_label = []
dev_data = []
dev_label = []
test_data = []
test_label = []

token_len = args.token_len

for i in tqdm(range(len(composer))):
    midi_data = pretty_midi.PrettyMIDI(f'./data/{args.task}.0.0/{midi_filename[i]}')
    pitch = get_pitch(midi_data, token_len)
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
                  
train_t = (torch.Tensor(train_data)+128).long()
dev_t = (torch.Tensor(dev_data)+128).long()
test_t = (torch.Tensor(test_data)+128).long()

# remove "/" in model name
model_replace = args.model
model_replace = model_replace.replace("/", "_")
print(model_replace)

model_process_path = f'./data/pkl/{model_replace}'




if not os.path.exists(model_process_path):
    os.makedirs(model_process_path)
    
torch.save(train_t, model_process_path + f'/{model_replace}_{token_len}_train_data.pkl')
torch.save(torch.LongTensor(train_label), model_process_path + f'/{model_replace}_{token_len}_train_label.pkl')
torch.save(dev_t, model_process_path + f'/{model_replace}_{token_len}_dev_data.pkl')
torch.save(torch.LongTensor(dev_label), model_process_path + f'/{model_replace}_{token_len}_dev_label.pkl')
torch.save(test_t, model_process_path + f'/{model_replace}_{token_len}_test_data.pkl')
torch.save(torch.LongTensor(test_label), model_process_path + f'/{model_replace}_{token_len}_test_label.pkl')
torch.save(composer2id, model_process_path + f'/composer2id_map.pkl')


# dev_data = []
# dev_label = []
# test_data = []
# test_label = []
# for i in tqdm(range(len(composer))):
#     if split[i] == 'train':
#         continue
#     midi_data = pretty_midi.PrettyMIDI(f'./{task}.0.0/{midi_filename[i]}')
#     pitch = get_pitch(midi_data)
#     label = composer2id[composer[i]]
#     if split[i] == 'validation':
#         dev_data.append((torch.Tensor(pitch)+128).long())  
#         dev_label.extend([label]*1)
#     elif split[i] == 'test':
#         test_data.append((torch.Tensor(pitch)+128).long())
#         test_label.extend([label]*1)
#     else:
#         raise NotImplementedError
        
# torch.save(dev_data, f'./data/{task}/{task}_bert-base-uncased_dev_data.pkl')
# torch.save(torch.LongTensor(dev_label), f'./data/{task}/{task}_bert-base-uncased_dev_label.pkl')
# torch.save(test_data, f'./data/{task}/{task}_bert-base-uncased_test_data.pkl')
# torch.save(torch.LongTensor(test_label), f'./data/{task}/{task}_bert-base-uncased_test_label.pkl')