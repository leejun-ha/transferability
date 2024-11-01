import os
import json
import pretty_midi
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import pandas as pd

# Set the path to the MAESTRO v1 dataset
MAESTRO_PATH = "/home/junha/transferability/Music/data/maestro-v1.0.0"

def get_train_files(metadata_path):
    try:
        csv = pd.read_csv(metadata_path)
        train_files = csv[csv['split'] == 'train']['midi_filename'].tolist()
        return train_files
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def analyze_midi(midi_file, tokens_to_remove=None):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tokens = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_on = f"NOTE_ON_{note.pitch}"
            note_off = f"NOTE_OFF_{note.pitch}"
            if tokens_to_remove is None or note_on not in tokens_to_remove:
                tokens.append(note_on)
            if tokens_to_remove is None or note_off not in tokens_to_remove:
                tokens.append(note_off)
    return tokens

def analyze_dataset(train_files, tokens_to_remove=None):
    all_tokens = []
    for midi_file in tqdm(train_files, desc="Analyzing MIDI files"):
        file_path = os.path.join(MAESTRO_PATH, midi_file)
        try:
            all_tokens.extend(analyze_midi(file_path, tokens_to_remove))
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    return all_tokens

def save_frequency_data(token_freq, output_file):
    with open(output_file, 'w') as f:
        json.dump(token_freq, f, indent=2)

def plot_frequency_graph(token_freq, output_file, title):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(token_freq)), list(token_freq.values()), align='center')
    plt.xticks(range(len(token_freq)), list(token_freq.keys()), rotation=90)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def load_frequency_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_tokens_to_remove(frequency_data):
    sorted_tokens = sorted(frequency_data.items(), key=lambda x: x[1])
    total_freq = sum(freq for _, freq in sorted_tokens)
    cumulative_freq = 0
    tokens_to_remove = set()

    for token, freq in sorted_tokens:
        cumulative_freq += freq
        tokens_to_remove.add(token)
        if cumulative_freq >= total_freq * 0.5:
            break

    return tokens_to_remove

def main():
    metadata_path = os.path.join(MAESTRO_PATH, 'maestro-v1.0.0.csv')
    train_files = get_train_files(metadata_path)
    
    if not train_files:
        print("No train files found. Exiting.")
        return

    # Analyze original train dataset
    print("Analyzing original train dataset...")
    all_tokens_original = analyze_dataset(train_files)
    token_freq_original = dict(Counter(all_tokens_original))
    token_freq_original = dict(sorted(token_freq_original.items(), key=lambda x: x[1], reverse=True))
    
    # Save original frequency data
    save_frequency_data(token_freq_original, 'maestro_v1_train_token_freq_original.json')
    plot_frequency_graph(token_freq_original, 'maestro_v1_train_token_freq_original.png', 'Token Frequency in MAESTRO v1 Train Set (Original)')
    
    # Load the test frequency data and get tokens to remove
    test_frequency_data = load_frequency_data('maestro_v1_test_token_freq.json')
    tokens_to_remove = get_tokens_to_remove(test_frequency_data)

    # Analyze train dataset with OOV tokens removed
    print("Analyzing train dataset with OOV tokens removed...")
    all_tokens_oov = analyze_dataset(train_files, tokens_to_remove)
    token_freq_oov = dict(Counter(all_tokens_oov))
    token_freq_oov = dict(sorted(token_freq_oov.items(), key=lambda x: x[1], reverse=True))
    
    # Save OOV frequency data
    save_frequency_data(token_freq_oov, 'maestro_v1_train_token_freq_oov.json')
    plot_frequency_graph(token_freq_oov, 'maestro_v1_train_token_freq_oov.png', 'Token Frequency in MAESTRO v1 Train Set (OOV Removed)')
    
    print("Analysis complete. Results saved as JSON and PNG files for both original and OOV-removed datasets.")

if __name__ == "__main__":
    main()