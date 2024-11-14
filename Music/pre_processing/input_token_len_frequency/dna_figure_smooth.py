import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Function to read token counts from a file and limit to token lengths between 0 and 512
def read_token_counts(filename):
    token_counts = defaultdict(int)
    with open(filename, 'r') as f:
        for line in tqdm(f, desc="Reading Token Counts", unit=" lines"):
            length, count = map(int, line.strip().split(': '))
            if 64 <= length <= 512:
                token_counts[length] += count
    return token_counts

# Function to load performance data from a JSON file
def load_performance_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def create_single_graph(model, ax=None, save_individual=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load performance data for the model
    performance_data = load_performance_data(f'{model}_result/performance_results.json')
    
    # Read token counts from the text file (limited to token lengths between 64 and 512)
    counts_file = f'{model}_result/token_counts_original.txt'
    token_counts = read_token_counts(counts_file)

    # Prepare data for plotting token lengths (64-512)
    all_lengths = list(token_counts.keys())
    all_counts = list(token_counts.values())
    
    # Plot token count as bars for lengths from 64 to 512
    # ax.bar(all_lengths, all_counts, color='lightsteelblue', width=1, label='Token Count (Bar)')
    
    # Plot specific token counts (64, 128, 256, 384, 512) as a line with markers
    specific_lengths = [64, 128, 256, 384, 512]
    specific_counts = [token_counts.get(length, 0) for length in specific_lengths]
    
    ax.plot(specific_lengths, specific_counts, color='blue', linewidth=2, marker='o', label='Token Count (Line)')
    
    # Annotate actual token count values on the graph for specific lengths
    for x_val, y_val in zip(specific_lengths, specific_counts):
        ax.annotate(f'{y_val}', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)
    
    # Set logarithmic scale for y-axis (token counts)
    ax.set_yscale('log')
    
    # Set common y-axis limits for token counts
    ax.set_ylim(common_y_count_min, common_y_count_max)
    
    # Set x-axis range from 64 to 512 with a margin on both sides
    ax.set_xlim(32, 544)  # Adding margin by extending beyond 64 and 512
    
    # Add labels and title
    ax.set_xlabel('Input Token Sequence Length', fontsize=24)
    ax.set_ylabel('Count (log scale)', fontsize=24)

    # Plot performance data (line chart on secondary y-axis)
    perf_lengths = [64, 128, 256, 384, 512]
    
    # Extract corresponding performance values from JSON data
    performances = [performance_data[str(length)] for length in perf_lengths]
    
    ax2 = ax.twinx()  # Create a secondary y-axis for performance data
    
    # Plot performance data on the same x-axis as token lengths but with its own y-axis
    line = ax2.plot(perf_lengths, performances, color='red', marker='o', label='Accuracy')
    
    for x_val_perf, y_val_perf in zip(perf_lengths, performances):
        ax2.annotate(f'{y_val_perf:.3f}', (x_val_perf, y_val_perf), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)
    
    # Set common y-axis limits for performance
    ax2.set_ylim(common_y_perf_min, common_y_perf_max)
    
    # Set labels for performance axis
    ax2.set_ylabel('Accuracy', fontsize=20, rotation=270, labelpad=25)

    # Combine legends from both axes (token count and performance)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=18)

    # Set x-axis ticks to show specific lengths (64-512)
    ax.set_xticks(specific_lengths)
    
    if save_individual:
        # Adjust layout and add margins to the left and right using subplots_adjust()
        plt.subplots_adjust(left=0.1, right=0.9)  # Adjust these values as needed
        
        plt.tight_layout()
        plt.savefig(f'figure/{model}_token_count_vs_performance.png', dpi=300,bbox_inches='tight')
        plt.close()
        print(f"Graph saved as '{model}_token_count_vs_performance.png'")

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

# Set common y-axis limits for token counts and performance
common_y_count_min = 1e0  # Minimum count (log scale)
common_y_count_max = 1e7  # Maximum count (log scale)

common_y_perf_min = 0.0      # Minimum performance value
common_y_perf_max = 0.7  # Maximum performance value

# Create individual graphs for each model
for model in models:
   create_single_graph(model)

# Create combined graph with subplots (for first five models only)
fig, axs = plt.subplots(2 ,3 ,figsize=(30 ,20))
fig.suptitle("Token Count Distribution and Performance for Different Models" ,fontsize=28)

for i in range(5):  
   row_idx=i//3  
   col_idx=i%3  
   create_single_graph(models[i],ax=axs[row_idx][col_idx],save_individual=False)

# Remove empty subplot if necessary 
fig.delaxes(axs[1][2])

plt.tight_layout()
plt.savefig('figure/combined_token_count_vs_performance.png' ,dpi=300,bbox_inches='tight')
plt.close()

print("Combined graph saved as 'combined_token_count_vs_performance.png'")
