import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Function to read token counts from a file and limit to token lengths between 0 and 1024
def read_token_counts(filename):
    token_counts = defaultdict(int)
    with open(filename, 'r') as f:
        for line in tqdm(f, desc="Reading Token Counts", unit=" lines"):
            length, count = map(int, line.strip().split(': '))
            if 0 <= length <= 512:  # Limit to token lengths between 0 and 512
                token_counts[length] += count
    return token_counts

# Function to load performance data from a JSON file
def load_performance_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# List of models
models = [
    'bert-base-uncased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'neuralmind_bert-base-portuguese-cased',
    'tohoku-nlp_bert-base-japanese',
    'microsoft_codebert-base-mlm',
    'neulab_codebert-javascript',
    'neulab_codebert-java',
    'neulab_codebert-python',
    'neulab_codebert-c'
]

# Create a figure with subplots for each model
fig, axs = plt.subplots(5, 2, figsize=(20, 30))
fig.suptitle("", fontsize=16)

# Set common y-axis limits for token counts and performance
common_y_count_min = 1e0  # Minimum count (log scale)
common_y_count_max = 1e5  # Maximum count (log scale)

common_y_perf_min = 0      # Minimum performance value
common_y_perf_max = 1      # Maximum performance value

for i, model in enumerate(models):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    
    # Load performance data for the model
    performance_data = load_performance_data(f'{model}_result/performance_results.json')
    
    # Read token counts from the text file (limited to token lengths between 0 and 1024)
    counts_file = f'{model}_result/token_counts_original.txt'
    token_counts = read_token_counts(counts_file)

    # Prepare data for plotting
    lengths = list(token_counts.keys())
    counts = list(token_counts.values())
    
    # Plot token count distribution (bar chart)
    ax.bar(lengths, counts, alpha=0.5, label='Token Count')
    
    # Set logarithmic scale for y-axis (token counts)
    ax.set_yscale('log')
    
    # Set common y-axis limits for token counts across all subplots
    ax.set_ylim(common_y_count_min, common_y_count_max)
    
    # Add labels and title for each subplot
    ax.set_xlabel('Input Token Length')
    ax.set_ylabel('Count (log scale)')
    ax.set_title(model)

    # Plot performance data (line chart on secondary y-axis)
    
    perf_lengths = [64, 128, 256, 384, 512]
    
    # Extract corresponding performance values from JSON data
    performances = [performance_data[str(length)] for length in perf_lengths]
    
    ax2 = ax.twinx()  # Create a secondary y-axis for performance data
    
    # Plot performance data on the same x-axis as token lengths but with its own y-axis
    ax2.plot(perf_lengths, performances, color='red', marker='o', label='Performance')
    
    # Set common y-axis limits for performance across all subplots
    ax2.set_ylim(common_y_perf_min, common_y_perf_max)
    
    # Set labels for performance axis
    ax2.set_ylabel('Performance')

    # Combine legends from both axes (token count and performance)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Adjust layout and save the final figure as an image file
plt.tight_layout()
plt.savefig('token_original_count_vs_performance_average_512.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph saved as 'token_count_vs_performance.png'")