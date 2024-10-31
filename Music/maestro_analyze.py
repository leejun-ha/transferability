import os
import json
import pretty_midi
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import pandas as pd

# Set the path to the MAESTRO v1 dataset
MAESTRO_PATH = "/home/junha/transferability/Music/data/maestro-v1.0.0"

def get_test_files(metadata_path):
    try:
        csv = pd.read_csv(metadata_path)
        test_files = csv[csv['split'] == 'test']['midi_filename'].tolist()
        return test_files
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
    return tokens

def analyze_dataset(test_files):
    all_tokens = []
    for midi_file in tqdm(test_files, desc="Analyzing MIDI files"):
        file_path = os.path.join(MAESTRO_PATH, midi_file)
        try:
            all_tokens.extend(analyze_midi(file_path))
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    return all_tokens

def save_frequency_data(token_freq, output_file):
    with open(output_file, 'w') as f:
        json.dump(token_freq, f, indent=2)

def plot_frequency_graph(token_freq, output_file):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(token_freq)), list(token_freq.values()), align='center')
    plt.xticks(range(len(token_freq)), list(token_freq.keys()), rotation=90)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('Token Frequency in MAESTRO v1 Test Set')
    plt.tight_layout()
    plt.savefig(output_file)

def main():
    metadata_path = os.path.join(MAESTRO_PATH, 'maestro-v1.0.0.csv')
    test_files = get_test_files(metadata_path)
    
    if not test_files:
        print("No test files found. Exiting.")
        return

    all_tokens = analyze_dataset(test_files)
    token_freq = dict(Counter(all_tokens))
    
    # Sort token frequency by count (descending order)
    token_freq = dict(sorted(token_freq.items(), key=lambda x: x[1], reverse=True))
    
    # Save frequency data as JSON
    save_frequency_data(token_freq, 'maestro_v1_test_token_freq.json')
    
    # Plot frequency graph
    plot_frequency_graph(token_freq, 'maestro_v1_test_token_freq.png')
    
    print("Analysis complete. Results saved as JSON and PNG files.")

if __name__ == "__main__":
    main()