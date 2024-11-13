import os
import json
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import pandas as pd

import numpy as np

# Set the path to the MAESTRO v1 dataset
MAESTRO_PATH = "/home/junha/transferability/Music/data/maestro-v1.0.0"

def get_all_files_and_composers(metadata_path):
    try:
        csv = pd.read_csv(metadata_path)
        return csv[['midi_filename', 'canonical_composer', 'split']].to_dict('records')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def analyze_midi(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tokens = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            tokens.append(f"NOTE_ON_{note.pitch}")
            tokens.append(f"NOTE_OFF_{note.pitch}")
    return tokens, midi_data.get_end_time(), len(tokens)

def analyze_dataset(files):
    all_tokens = []
    composer_counts = Counter()
    split_counts = Counter()
    lengths = []
    token_lengths = []
    for file_info in tqdm(files, desc="Analyzing MIDI files"):
        file_path = os.path.join(MAESTRO_PATH, file_info['midi_filename'])
        try:
            tokens, length, token_length = analyze_midi(file_path)
            all_tokens.extend(tokens)
            composer_counts[file_info['canonical_composer']] += 1
            split_counts[file_info['split']] += 1
            lengths.append(length)
            token_lengths.append(token_length)
        except Exception as e:
            print(f"Error processing {file_info['midi_filename']}: {e}")
    return all_tokens, composer_counts, split_counts, lengths, token_lengths

def save_frequency_data(data, output_file):
    with open(output_file, 'w') as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        elif isinstance(data, list):
            for item in data:
                f.write(f"{item}\n")
        else:
            f.write(str(data))

def plot_frequency_graph(data, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()), rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)

def main():
    metadata_path = os.path.join(MAESTRO_PATH, 'maestro-v1.0.0.csv')
    all_files = get_all_files_and_composers(metadata_path)
    
    if not all_files:
        print("No files found. Exiting.")
        return

    all_tokens, composer_counts, split_counts, lengths, token_lengths = analyze_dataset(all_files)
    token_freq = dict(Counter(all_tokens))
    
    # Calculate average MIDI length and range
    average_length = np.mean(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    
    # Calculate average token sequence length and range
    average_token_length = np.mean(token_lengths)
    min_token_length = np.min(token_lengths)
    max_token_length = np.max(token_lengths)
    
    # Sort token frequency by count (descending order)
    token_freq = dict(sorted(token_freq.items(), key=lambda x: x[1], reverse=True))
    
    # Sort composer counts by count (descending order)
    composer_counts = dict(sorted(composer_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Save frequency data as text files
    save_frequency_data(token_freq, 'maestro_v1_all_token_freq.txt')
    save_frequency_data(composer_counts, 'maestro_v1_all_composer_dist.txt')
    save_frequency_data(split_counts, 'maestro_v1_split_dist.txt')
    
    # Save MIDI length and token length statistics
    length_stats = {
        'average_length': float(average_length),
        'min_length': float(min_length),
        'max_length': float(max_length),
        'average_token_length': float(average_token_length),
        'min_token_length': int(min_token_length),
        'max_token_length': int(max_token_length)
    }
    save_frequency_data(length_stats, 'maestro_v1_length_stats.txt')
    
    # Plot frequency graphs
    plot_frequency_graph(token_freq, 'Token Frequency in MAESTRO v1 (All Splits)', 'Tokens', 'Frequency', 'maestro_v1_all_token_freq.png')
    plot_frequency_graph(composer_counts, 'Composer Distribution in MAESTRO v1 (All Splits)', 'Composers', 'Number of Pieces', 'maestro_v1_all_composer_dist.png')
    plot_frequency_graph(split_counts, 'Split Distribution in MAESTRO v1', 'Splits', 'Number of Files', 'maestro_v1_split_dist.png')
    
    print("Analysis complete. Results saved as JSON and PNG files.")
    print(f"Number of unique composers in the dataset: {len(composer_counts)}")
    print("Split distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")
    print(f"Average MIDI length: {average_length:.2f} seconds")
    print(f"MIDI length range: {min_length:.2f} to {max_length:.2f} seconds")
    print(f"Average token sequence length: {average_token_length:.2f} tokens")
    print(f"Token sequence length range: {min_token_length} to {max_token_length} tokens")

if __name__ == "__main__":
    main()