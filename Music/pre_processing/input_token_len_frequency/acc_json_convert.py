import os
import json

def read_combined_accuracy(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'Combined accuracy:' in line:
                return float(line.split(':')[-1].strip())
    return None

def read_average_accuracy(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'Average accuracy:' in line:
                return float(line.split(':')[-1].strip())
    return None
# Define the models and token lengths
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

token_lengths = [64, 128, 256, 384, 512]

# Initialize performance results dictionary


# Read combined accuracy values from each model's corresponding files
for model in models:
    performance_results = {str(length): 0.0 for length in token_lengths}
    model_performance = {}
    for length in token_lengths:
        # Construct the filename
        file_name = f'acc/{model.replace("/", "_")}_accuracy_results_test_maestro-v1_pretrain_seed2020_tokenlen{length}.txt'
        if os.path.exists(file_name):
            # combined_accuracy = read_combined_accuracy(file_name)
            combined_accuracy = read_average_accuracy(file_name)
            if combined_accuracy is not None:
                model_performance[str(length)] = combined_accuracy

    # Update the overall performance results
    for length, accuracy in model_performance.items():
        performance_results[length] = max(performance_results[length], accuracy)

    # Write the performance results to a JSON file
    with open(f'{model.replace("/", "_")}_result/performance_results.json', 'w') as f:
        json.dump(performance_results, f, indent=4)

print("Performance results have been saved to performance_results.json")
print("Final performance results:")
print(json.dumps(performance_results, indent=4))