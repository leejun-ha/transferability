import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

# Function to read token counts from a file and limit to token lengths between 64 and 512
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

# Function to create and save a graph for a single model
def create_single_graph(model, ax=None, save_individual=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load performance data for the model
    performance_data = load_performance_data(f'{model}_result/performance_results.json')
    
    # Read token counts from the text file (limited to token lengths between 64 and 512)
    counts_file = f'{model}_result/token_counts_original.txt'
    token_counts = read_token_counts(counts_file)

    # Prepare data for plotting
    lengths = list(token_counts.keys())
    counts = list(token_counts.values())
    
    # Fit a normal distribution to the token counts
    mu, std = norm.fit(np.repeat(lengths, counts))
    
    # Generate points for the fitted normal distribution
    x = np.linspace(64, 512, 100)
    p = norm.pdf(x, mu, std)
    p = p * (max(counts) / max(p))  # Scale the distribution to match the max count
    
    # Plot token count distribution (line plot)
    ax.plot(x, p, color='blue', alpha=0.7, label='Token Count Trend')
    
    # Set logarithmic scale for y-axis (token counts)
    ax.set_yscale('log')
    
    # Set common y-axis limits for token counts
    ax.set_ylim(common_y_count_min, common_y_count_max)
    
    # Add labels and title
    ax.set_xlabel('Input Token Length', fontsize=24)
    ax.set_ylabel('Count (log scale)', fontsize=24, color='blue')
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
    ax2.set_ylabel('Performance', fontsize=20, color='red')

    # Combine legends from both axes (token count and performance)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=18)

    # Set x-axis ticks to show 64, 128, 256, 384, 512
    ax.set_xticks(perf_lengths)
    ax.set_xticklabels(perf_lengths)

    # Set color for y-axis ticks and label
    ax.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='red')

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
common_y_count_min = 1e0  # Minimum count (log scale)
common_y_count_max = 1e4  # Maximum count (log scale)

common_y_perf_min = 0      # Minimum performance value
common_y_perf_max = 0.8    # Maximum performance value

# Create individual graphs
for model in models:
    create_single_graph(model)

# Create combined graph with 5 subplots
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
fig.suptitle("Token Count Trend vs Performance for Different Models", fontsize=28)

for i, model in enumerate(models[:5]):  # Only use the first 5 models
    row = i // 3
    col = i % 3
    create_single_graph(model, ax=axs[row, col], save_individual=False)

# Remove the empty subplot
fig.delaxes(axs[1, 2])

# Adjust layout and save the combined figure
plt.tight_layout()
plt.savefig('figure/combined_token_count_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Combined graph saved as 'combined_token_count_vs_performance.png'")
print("All graphs have been saved.")
