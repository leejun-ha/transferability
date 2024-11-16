import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def read_token_counts(filename):
    token_counts = defaultdict(int)
    with open(filename, 'r') as f:
        for line in f:
            length, count = map(int, line.strip().split(': '))
            token_counts[length] += count
    return token_counts

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

# Load performance data


# Create a figure with subplots for each model
fig, axs = plt.subplots(5, 2, figsize=(20, 30))
fig.suptitle("Token Count Distribution vs Fine-tuning Performance", fontsize=16)

for i, model in enumerate(models):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    performance_data = load_performance_data(f'{model}_result/performance_results.json')
    # Read token counts
    counts_file = f'{model}_result/token_counts_truncated.txt'
    token_counts = read_token_counts(counts_file)

    # Plot token count distribution
    lengths = list(token_counts.keys())
    counts = list(token_counts.values())
    ax.bar(lengths, counts, alpha=0.5, label='Token Count')
    ax.set_yscale('log')
    ax.set_xlabel('Input Token Length')
    ax.set_ylabel('Count (log scale)')
    ax.set_title(model)

    # Plot performance data
    perf_lengths = [64, 128, 256, 384, 512]
    performances = [performance_data[str(length)] for length in perf_lengths]
    ax2 = ax.twinx()
    ax2.plot(perf_lengths, performances, color='red', marker='o', label='Performance')
    ax2.set_ylabel('Performance')
    ax2.set_ylim(0, 1)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('token_count_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph saved as 'token_count_vs_performance.png'")