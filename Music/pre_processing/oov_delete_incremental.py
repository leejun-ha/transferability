import pretty_midi
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='maestro-v1.0.0')
parser.add_argument('--data_dir', type=str, default='../data/maestro-v1.0.0')
parser.add_argument('--save_dir', type=str, default='../data')
parser.add_argument('--model', type=str)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--token_len', type=int, default=128)
parser.add_argument('--json_file', type=str, default='maestro_v1_test_token_freq.json')
args = parser.parse_args()

def get_pitch_skip(midi_data, token_len, tokens_to_remove):
    note_list = []
    instrument = midi_data.instruments
    notes = instrument[0].notes
    current_segment = []

    for note in notes:
        if note.pitch not in tokens_to_remove:
            current_segment.append(note.pitch)
            
            if len(current_segment) == token_len:
                note_list.append(current_segment)
                current_segment = []

    return note_list

def get_pitch_mask(midi_data, token_len, tokens_to_remove, mask_token):
    note_list = []
    instrument = midi_data.instruments
    notes = instrument[0].notes
    current_segment = []

    for note in notes:
        if note.pitch in tokens_to_remove:
            current_segment.append(mask_token)
        else:
            current_segment.append(note.pitch)
        
        if len(current_segment) == token_len:
            note_list.append(current_segment)
            current_segment = []

    return note_list

def load_frequency_data(json_file):
    with open(json_file, 'r') as f:
        frequency_data = json.load(f)
    
    sorted_tokens = sorted([(int(token.split('_')[-1]), freq) for token, freq in frequency_data.items() if token.startswith('NOTE_ON')], key=lambda x: x[1])
    return [token for token, _ in sorted_tokens]

csv = pd.read_csv(f'{args.data_dir}/{args.task}.csv')

composer = csv['canonical_composer']
split = csv['split']
midi_filename = csv['midi_filename']
composer2id = {comp: idx for idx, comp in enumerate(set(composer))}

print("composer_num: ", len(composer2id))

# Load and process the frequency data
sorted_tokens = load_frequency_data(args.json_file)

token_len = args.token_len
mask_token = -1  # You can choose any value that's not used as a pitch

for version in range(1, 45):  # Create 44 versions
    tokens_to_remove = set(sorted_tokens[:version])
    
    for method in ['skip', 'mask']:
        train_data = []
        train_label = []
        dev_data = []
        dev_label = []
        test_data = []
        test_label = []

        for i in tqdm(range(len(composer)), desc=f"Processing version {version} ({method})"):
            midi_data = pretty_midi.PrettyMIDI(f'{args.data_dir}/{midi_filename[i]}')
            if method == 'skip':
                pitch = get_pitch_skip(midi_data, token_len, tokens_to_remove)
            else:  # mask
                pitch = get_pitch_mask(midi_data, token_len, tokens_to_remove, mask_token)
            
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

        # Convert to tensors
        train_t = (torch.Tensor(train_data)+128).long()
        dev_t = (torch.Tensor(dev_data)+128).long()
        test_t = (torch.Tensor(test_data)+128).long()

        # Save the data
        model_process_path = f'{args.save_dir}/pkl/version_{version}_{method}'
        os.makedirs(model_process_path, exist_ok=True)

        torch.save(train_t, f'{model_process_path}/{token_len}_train_data_filtered_inc.pkl')
        torch.save(torch.LongTensor(train_label), f'{model_process_path}/{token_len}_train_label_filtered_inc.pkl')
        torch.save(dev_t, f'{model_process_path}/{token_len}_dev_data_filtered_inc.pkl')
        torch.save(torch.LongTensor(dev_label), f'{model_process_path}/{token_len}_dev_label_filtered_inc.pkl')
        torch.save(test_t, f'{model_process_path}/{token_len}_test_data_filtered_inc.pkl')
        torch.save(torch.LongTensor(test_label), f'{model_process_path}/{token_len}_test_label_filtered_inc.pkl')
        torch.save(composer2id, f'{model_process_path}/composer2id_map_filtered_inc.pkl')

        print(f"Version {version} ({method}) - Processed data shapes:")
        print(f"Train: {train_t.shape}")
        print(f"Dev: {dev_t.shape}")
        print(f"Test: {test_t.shape}")
        print(f"Tokens removed: {len(tokens_to_remove)}")
        print()