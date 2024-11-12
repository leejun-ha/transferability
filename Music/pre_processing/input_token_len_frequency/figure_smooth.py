import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d  # For Gaussian smoothing

# Function to read token counts from a file and limit to token lengths between 64 and 512
def read_token_counts(filename):
    token_counts = defaultdict(int)
    with open(filename, 'r') as f:
        for line in tqdm(f, desc="Reading Token Counts", unit=" lines"):
            length, count = map(int, line.strip().split(': '))
            if length in [64, 128, 256, 384, 512]:  # Only consider representative values
                token_counts[length] += count
    return token_counts

# Function to load performance data from a JSON file
def load_performance_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Function to create and save a graph for a single model with Gaussian smoothing applied only to representative values
def create_single_graph(model, ax=None, save_individual=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load performance data for the model
    performance_data = load_performance_data(f'{model}_result/performance_results.json')
    
    # Read token counts from the text file (limited to representative values: 64, 128, 256, 384, and 512)
    counts_file = f'{model}_result/token_counts_original.txt'
    token_counts = read_token_counts(counts_file)

    # Prepare data for plotting
    lengths = np.array([64, 128, 256, 384, 512])  # Representative lengths
    counts = np.array([token_counts[length] for length in lengths])  # Corresponding counts
    
    # Apply Gaussian smoothing to the token count data (for the representative values)
    counts_smooth = gaussian_filter1d(counts, sigma=1)  # Adjust sigma for more or less smoothing
    
    # Plot only the smoothed token count distribution (line plot)
    ax.plot(lengths, counts_smooth, color='blue', alpha=0.7, marker='o', label='Token Count')
    
    # Set logarithmic scale for y-axis (token counts)
    ax.set_yscale('log')
    
    # Set common y-axis limits for token counts
    ax.set_ylim(common_y_count_min, common_y_count_max)
    
    # Add labels and title
    ax.set_xlabel('Input Token Length', fontsize=24)
    ax.set_ylabel('Count (log scale)', fontsize=24)
    ax.set_title(model, fontsize=24)

    # Plot performance data (line chart on secondary y-axis)
    perf_lengths = [64, 128, 256, 384, 512]
    
    # Extract corresponding performance values from JSON data
    performances = [performance_data[str(length)] for length in perf_lengths]
    
    ax2 = ax.twinx()  # Create a secondary y-axis for performance data
    
    # Plot performance data on the same x-axis as token lengths but with its own y-axis
    line = ax2.plot(perf_lengths, performances, color='red', marker='o', label='Performance')
    
    for x, y in zip(perf_lengths, performances):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)
    
    # Set common y-axis limits for performance
    ax2.set_ylim(common_y_perf_min, common_y_perf_max)
    
    # Set labels for performance axis
    ax2.set_ylabel('Performance', fontsize=20)

    # Combine legends from both axes (token count and performance)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=18)

    # Set x-axis ticks to show only the representative values: 64, 128, 256, 384, and 512
    ax.set_xticks(perf_lengths)
    ax.set_xticklabels(perf_lengths)

    # Set color for y-axis ticks and label
    ax.tick_params(axis='y')
    ax2.tick_params(axis='y')

    if save_individual:
        # Adjust layout and save the figure as an individual PNG file
        plt.tight_layout()
        plt.savefig(f'figure/{model}_token_count_vs_performance.png', dpi=300, bbox_inches='tight')
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
common_y_count_min = 1e0   # Minimum count (log scale)
common_y_count_max = 1e6   # Maximum count (log scale)

common_y_perf_min = 0      # Minimum performance value
common_y_perf_max = 0.8     # Maximum performance value

# Create individual graphs for each model in the list of models
for model in models:
    create_single_graph(model)

# Create combined graph with subplots for multiple models (first five models in this case)
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
fig.suptitle("Token Count vs Performance for Different Models", fontsize=28)

for i, model in enumerate(models[:5]):   # Only use the first five models
   row = i // 3
   col = i % 3
   create_single_graph(model, ax=axs[row][col], save_individual=False)

# Remove the empty subplot if there are fewer than six subplots needed.
fig.delaxes(axs[1][2])

# Adjust layout and save the combined figure as an image file.
plt.tight_layout()
plt.savefig('figure/combined_token_count_vs_performance.png', dpi=300,
            bbox_inches='tight')
plt.close()

print("Combined graph saved as 'combined_token_count_vs_performance.png'")
print("All graphs have been saved.")
