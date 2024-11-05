import pretty_midi
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='maestro-v1')
parser.add_argument('--data_dir', type=str, default='./data/maestro-v1.0.0')
parser.add_argument('--save_dir', type=str, default='./data')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--token_len', type=int, default=128)
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
        note = [n.pitch for n in note]
        note_list.append(note)
        i += 1
    return note_list

all_data = []
all_label = []

token_len = args.token_len

for i in tqdm(range(len(composer))):
    midi_data = pretty_midi.PrettyMIDI(f'{args.data_dir}/{midi_filename[i]}')
    pitch = get_pitch(midi_data, token_len)
    label = composer2id[composer[i]]
    
    all_data.extend(pitch)
    all_label.extend([label] * len(pitch))

# Convert to tensors

all_data_t = (torch.Tensor(all_data) + 128).long()



model_process_path = f'{args.save_dir}/pkl'

if not os.path.exists(model_process_path):
    os.makedirs(model_process_path)

# Save all data in a single pkl file
all_data_file = f'{model_process_path}/{token_len}_all_data.pkl'
torch.save(all_data_t, all_data_file)
print(f"All data saved to {all_data_file}")

# Save all labels in a single pkl file
all_labels_file = f'{model_process_path}/{token_len}_all_label.pkl'
torch.save(torch.LongTensor(all_label), all_labels_file)
print(f"All labels saved to {all_labels_file}")

# Save composer2id mapping
composer2id_file = f'{model_process_path}/composer2id_map.pkl'
torch.save(composer2id, composer2id_file)
print(f"Composer to ID mapping saved to {composer2id_file}")