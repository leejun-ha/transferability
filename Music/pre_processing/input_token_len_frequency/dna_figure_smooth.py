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
    
    ax.plot(specific_lengths, specific_counts, color='blue', linewidth=2, marker='o', label='Sequence Count')
    
    # Annotate actual token count values on the graph for specific lengths
    for x_val, y_val in zip(specific_lengths, specific_counts):
        ax.annotate(f'{y_val}', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=20)
    
    # Set logarithmic scale for y-axis (token counts)
    ax.set_yscale('log')
    
    # Set common y-axis limits for token counts
    ax.set_ylim(common_y_count_min, common_y_count_max)
    
    # Set x-axis range from 64 to 512 with a margin on both sides
    ax.set_xlim(32, 544)  # Adding margin by extending beyond 64 and 512
    
    # Add labels and title
    ax.set_xlabel('Input Token Sequence Length', fontsize=26)
    ax.set_ylabel('Count (log scale)', fontsize=26)

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
    ax2.set_ylabel('Accuracy', fontsize=24, rotation=270, labelpad=25)

    # Combine legends from both axes (token count and performance)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=18)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax2.tick_params(axis='y', which='major', labelsize=26)

    # Set x-axis ticks to show specific lengths (64-512)
    ax.set_xticks(specific_lengths)
    
    if save_individual:
        # Adjust layout and add margins to the left and right using subplots_adjust()
        plt.subplots_adjust(left=0.1, right=0.9)  # Adjust these values as needed
        
        plt.tight_layout()
        plt.savefig(f'figure/{model}_token_count_vs_performance.png', dpi=300,bbox_inches='tight')
        plt.close()
        print(f"Graph saved as '{model}_token_count_vs_performance.png'")

def create_combined_graph(models, paper_name_models, output_filename):
    fig, axs = plt.subplots(2, 1, figsize=(15, 20))
    # fig.suptitle("Token Count Distribution and Performance Comparison", fontsize=28)

    for i, model in enumerate(models):
        create_single_graph(model, ax=axs[i], save_individual=False)
    
    for i, model_name in enumerate(paper_name_models):
        axs[i].set_title(model_name, fontsize=22)
    
    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)

    plt.savefig(f'figure/{output_filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined graph saved as '{output_filename}'")

def create_combined_graph_2(model_pairs, output_filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    # fig.suptitle("Token Count Distribution and Performance Comparison", fontsize=28)

    # Global y-axis limits
    common_y_count_min = 1e1
    common_y_count_max = 1e7
    common_y_perf_min = 0.1
    common_y_perf_max = 0.7

    # Specific token lengths
    specific_lengths = [64, 128, 256, 384, 512]

    for ax, (model, model_title) in zip([ax1, ax2], model_pairs):
        performance_data = load_performance_data(f'{model}_result/performance_results.json')
        counts_file = f'{model}_result/token_counts_original.txt'
        token_counts = read_token_counts(counts_file)

        specific_counts = [token_counts.get(length, 0) for length in specific_lengths]
        performances = [performance_data[str(length)] for length in specific_lengths]

        # Main plot for token counts
        ax.plot(specific_lengths, specific_counts, color='blue', linewidth=2, marker='o', label='Sequence Count')
        ax.set_yscale('log')
        ax.set_ylim(common_y_count_min, common_y_count_max)
        ax.set_xlim(32, 544)
        ax.set_xlabel('Input Token Sequence Length', fontsize=26)
        ax.set_ylabel('Count (log scale)', fontsize=26)
        ax.set_title(model_title, fontsize=26)  # Use the paper name model title

        # Secondary y-axis for performance
        ax_perf = ax.twinx()
        ax_perf.plot(specific_lengths, performances, color='red', marker='o', label='Accuracy')
        ax_perf.set_ylim(common_y_perf_min, common_y_perf_max)
        ax_perf.set_ylabel('Accuracy', fontsize=24, rotation=270, labelpad=25)

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_perf.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=18)

        # Annotate values
        for x, y, p in zip(specific_lengths, specific_counts, performances):
            ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=26)
            ax_perf.annotate(f'{p:.3f}', (x, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=26)

        # Set x-axis ticks
        ax.set_xticks(specific_lengths)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax_perf.tick_params(axis='y', which='major', labelsize=26)

    plt.tight_layout()
    plt.savefig(f'figure/{output_filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined graph saved as '{output_filename}'")

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

paper_name_models = [
    'BERT (English)',
    'BERT-c',
    'BERT-g',
    'BERT-p',
    'BERT-j',
    'CodeBERT',
    'Javascript',
    'Java',
    'Python',
    'C',
]

# Create combined graphs with 2 models per figure
# Create a list of tuples pairing each model with its paper name
model_pairs = list(zip(models, paper_name_models))

# Create combined graphs with 2 models per figure
for i in range(0, len(model_pairs), 2):
    create_combined_graph_2(
        model_pairs[i:i+2],
        f'combined_token_count_vs_performance_{i//2+1}.png'
    )
# Set common y-axis limits for token counts and performance
common_y_count_min = 1e1  # Minimum count (log scale)
common_y_count_max = 1e7  # Maximum count (log scale)

common_y_perf_min = 0.0    # Minimum performance value
common_y_perf_max = 0.7  # Maximum performance value

# Create individual graphs for each model
for model in models:
   create_single_graph(model)



plt.tight_layout()
plt.savefig('figure/combined_token_count_vs_performance.png' ,dpi=300,bbox_inches='tight')
plt.close()

print("Combined graph saved as 'combined_token_count_vs_performance.png'")
