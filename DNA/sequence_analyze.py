import matplotlib.pyplot as plt
from collections import Counter
import re

def parse_fasta(file_path):
    sequences = {}
    current_id = ""
    current_sequence = ""
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = current_sequence
                current_id = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
    
    if current_id:
        sequences[current_id] = current_sequence
    
    return sequences

def count_sequence_lengths(sequences):
    return {id: len(seq) for id, seq in sequences.items()}

def save_results(lengths, output_file):
    with open(output_file, 'w') as f:
        for id, length in lengths.items():
            f.write(f"{id}: {length}\n")

def plot_distribution(lengths, output_file, title):
    plt.figure(figsize=(12, 6))
    plt.hist(list(lengths.values()), bins=50, edgecolor='black')
    plt.title(f'Distribution of Sequence Lengths - {title}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.savefig(output_file)
    plt.close()

def analyze_sequences(file_path, output_prefix):
    sequences = parse_fasta(file_path)
    sequence_lengths = count_sequence_lengths(sequences)

    # Save results as txt file
    save_results(sequence_lengths, f'{output_prefix}_sequence_lengths.txt')

    # Create and save distribution plot
    plot_distribution(sequence_lengths, f'{output_prefix}_length_distribution.png', output_prefix)

    print(f"Analysis complete for {output_prefix}. Results saved in '{output_prefix}_sequence_lengths.txt' and '{output_prefix}_length_distribution.png'.")

def main():
    h4_file = '/home/junha/transferability/DNA/Hilbert-CNN/data/H4.txt'  # Replace with your H4 FASTA file path
    h3k9ac_file = '/home/junha/transferability/DNA/Hilbert-CNN/data/H3K9ac.txt'  # Replace with your H3K9ac FASTA file path

    analyze_sequences(h4_file, 'H4')
    analyze_sequences(h3k9ac_file, 'H3K9ac')

if __name__ == "__main__":
    main()