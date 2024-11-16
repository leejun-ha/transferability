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
    
    # Select top, middle, and low 128 tokens
    total_tokens = len(sorted_tokens)
    top_128 = sorted_tokens[:128]
    middle_128 = sorted_tokens[total_tokens//2 - 64:total_tokens//2 + 64]
    low_128 = sorted_tokens[-128:]
    
    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot token frequency distribution for selected tokens
    x = range(128)
    
    ax1.plot(x, [freq for _, freq in top_128], color='blue', alpha=0.5, label='Top 128')
    ax1.plot(x, [freq for _, freq in middle_128], color='green', alpha=0.5, label='Middle 128')
    ax1.plot(x, [freq for _, freq in low_128], color='orange', alpha=0.5, label='Low 128')
    
    ax1.set_xlabel('Token Rank (within each group)')
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')
    
    # Create secondary axis for accuracy
    ax2 = ax1.twinx()
    
    # Plot accuracy bars
    bar_positions = [32, 64, 96]  # Positions within the 128-token range
    bar_values = [frequency_results['top'], frequency_results['middle'], frequency_results['low']]
    bar_colors = ['red', 'green', 'orange']
    bar_labels = ['Top', 'Middle', 'Low']
    
    bars = ax2.bar(bar_positions, bar_values, width=32, alpha=0.3, color=bar_colors)
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