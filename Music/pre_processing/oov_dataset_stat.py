import os
import pandas as pd
import torch
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/pkl')
args = parser.parse_args()

# Initialize token count dictionary and token occurrence dictionary
token_counts = {}
token_occurrences = defaultdict(list)

# Define datasets
datasets = ['all']

# Process each dataset
total_sequences = 0
for dataset in datasets:
    try:
        data_path = os.path.join(args.data_dir, f'256_{dataset}_data.pkl')
        data = torch.load(data_path)
        
        dataset_offset = total_sequences
        total_sequences += data.shape[0]
        
        # Count sequences containing each token and record occurrences
        for seq_idx, sequence in enumerate(data):
            unique_tokens = torch.unique(sequence)
            for token in unique_tokens:
                token_item = token.item()
                if token_item not in token_counts:
                    token_counts[token_item] = 1
                else:
                    token_counts[token_item] += 1
                token_occurrences[token_item].append(dataset_offset + seq_idx)
        
    except FileNotFoundError:
        print(f"Data for {dataset} dataset not found.")
    except Exception as e:
        print(f"An error occurred while processing {dataset} dataset: {str(e)}")

# Create a list of dictionaries for the DataFrame
statistics = [
    {
        'token': token,
        'sequence_count': count,
        'percentage': (count / total_sequences) * 100,
        'occurrences': token_occurrences[token]
    } 
    for token, count in token_counts.items()
]

# Create DataFrame from statistics
statistics_df = pd.DataFrame(statistics)

# Check if the DataFrame is empty
if statistics_df.empty:
    print("No data to process. The DataFrame is empty.")
else:
    # Sort the DataFrame by sequence_count in descending order
    statistics_df = statistics_df.sort_values('sequence_count', ascending=False)

    # Display the results (excluding the 'occurrences' column for readability)
    print(statistics_df[['token', 'sequence_count', 'percentage']])

    # Optionally, save the statistics to a CSV file
    try:
        output_file = 'token_statistics_bert_base_uncased_with_occurrences.csv'
        statistics_df.to_csv(output_file, index=False)
        print(f"Statistics saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {str(e)}")

print(f"Total number of sequences processed: {total_sequences}")