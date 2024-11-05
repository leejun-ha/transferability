import json
import matplotlib.pyplot as plt
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_frequency_ranking(model_name, token_frequency, frequency_results):
    total_tokens = len(token_frequency)
    third = total_tokens // 3

    top = list(token_frequency.keys())[:third]
    middle = list(token_frequency.keys())[third:2*third]
    low = list(token_frequency.keys())[2*third:]

    # Bar chart
    categories = ['Top', 'Middle', 'Low']
    performance = [frequency_results['top'], frequency_results['middle'], frequency_results['low']]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, performance)
    plt.title(f'Performance by Token Frequency Category - {model_name}')
    plt.xlabel('Token Frequency Category')
    plt.ylabel('Performance')
    plt.ylim(0, 1)
    for i, v in enumerate(performance):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    plt.savefig(f'{model_name}_performance_bar_chart.png')
    plt.close()

    # Line chart
    top_128 = list(token_frequency.values())[:128]
    middle_128 = list(token_frequency.values())[third:third+128]
    low_128 = list(token_frequency.values())[-128:]

    plt.figure(figsize=(12, 6))
    plt.plot(range(128), top_128, label='Top 128')
    plt.plot(range(128), middle_128, label='Middle 128')
    plt.plot(range(128), low_128, label='Low 128')
    plt.title(f'Token Frequency Distribution - {model_name}')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{model_name}_frequency_line_chart.png')
    plt.close()

    print(f"Charts for {model_name} have been saved.")

# List of models
models = [
    "bert-base-uncased",
    "bert-base-chinese",
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
        compare_frequency_ranking(model.replace('/', '_'), token_frequency, frequency_results)
    else:
        print(f"Required files not found for {model}")

print("Processing complete for all models.")