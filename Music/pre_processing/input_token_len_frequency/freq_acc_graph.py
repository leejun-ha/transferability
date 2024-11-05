import json
import matplotlib.pyplot as plt
import os
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_overlapped_chart(model_name, token_frequency, frequency_results, y_min=0.5, y_max=0.6):
    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_frequency.items(), key=lambda x: x[1], reverse=True)
    
    # Split tokens into top, middle, and low
    total_tokens = len(sorted_tokens)
    split_point = total_tokens // 3
    
    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot token frequency distribution
    x = range(len(sorted_tokens))
    y = [freq for _, freq in sorted_tokens]
    ax1.plot(x, y, color='blue', alpha=0.5, label='Token Frequency')
    ax1.set_xlabel('Tokens (sorted by frequency)')
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')
    
    # Create secondary axis for accuracy
    ax2 = ax1.twinx()
    
    # Plot accuracy bars
    bar_positions = [split_point/2, split_point*1.5, split_point*2.5]
    bar_values = [frequency_results['top'], frequency_results['middle'], frequency_results['low']]
    bar_colors = ['red', 'green', 'orange']
    bar_labels = ['Top', 'Middle', 'Low']
    
    bars = ax2.bar(bar_positions, bar_values, width=split_point, alpha=0.3, color=bar_colors)
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(y_min, y_max)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    # Add legend
    ax1.legend(loc='upper right')
    ax2.legend(bars, bar_labels, loc='lower right')
    
    # Add vertical lines to separate top, middle, and low sections
    ax1.axvline(x=split_point, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=2*split_point, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'Token Frequency Distribution vs Model Accuracy - {model_name}')
    plt.tight_layout()
    
    # Save the figure in the model's directory
    save_path = os.path.join(f"{model_name}_result", f"{model_name}_frequency_vs_accuracy_chart.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Chart for {model_name} has been saved in its result directory.")

# List of models
models = [
    "bert-base-uncased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "neuralmind/bert-base-portuguese-cased",
    "tohoku-nlp/bert-base-japanese",
    "microsoft/codebert-base-mlm",
    "neulab/codebert-javascript",
    "neulab/codebert-java",
    "neulab/codebert-python",
    "neulab/codebert-c"
]

# Process each model
for model in models:
    model_dir = f"{model.replace('/', '_')}_result"
    token_frequency_path = os.path.join(model_dir, "token_frequency_ranking.json")
    frequency_results_path = os.path.join(model_dir, "frequency_results.json")

    if os.path.exists(token_frequency_path) and os.path.exists(frequency_results_path):
        token_frequency = load_json(token_frequency_path)
        frequency_results = load_json(frequency_results_path)
        
        # Find the min and max accuracy values to set y-axis limits
        accuracies = [frequency_results['top'], frequency_results['middle'], frequency_results['low']]
        y_min = min(accuracies) - 0.05  # Add some padding
        y_max = max(accuracies) + 0.05  # Add some padding
        
        create_overlapped_chart(model.replace('/', '_'), token_frequency, frequency_results, y_min, y_max)
    else:
        print(f"Required files not found for {model}")

print("Processing complete for all models.")