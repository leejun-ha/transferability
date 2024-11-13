import torch
import json
import os
from collections import OrderedDict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_sorted_tokens(freq_dict):
    return sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

def create_percentile_mapping(tokens):
    total = len(tokens)
    return {token: i / total for i, (token, _) in enumerate(tokens)}

def process_model(model_name, maestro_freq):
    # Load BERT-like model frequency data
    model_replace = model_name.replace("/", "_")
    bert_freq_path = f'./{model_replace}_result/token_frequency_ranking.json'
    bert_freq = load_json(bert_freq_path)

    # Sort tokens by frequency
    bert_tokens = get_sorted_tokens(bert_freq)
    maestro_tokens = get_sorted_tokens(maestro_freq)

    # Create percentile mappings
    bert_percentiles = create_percentile_mapping(bert_tokens)
    maestro_percentiles = create_percentile_mapping(maestro_tokens)

    # Create the mapping tensor
    mapping_tensor = torch.zeros(256, 1, dtype=torch.long)
    
    for i in range(256):
        if i < 128:
            mapping_tensor[i, 0] = i  # Keep original mapping for 0-127
        else:
            maestro_index = i - 128
            if maestro_index < len(maestro_tokens):
                maestro_token, _ = maestro_tokens[maestro_index]
                maestro_percentile = maestro_percentiles[maestro_token]
                
                # Find the closest BERT token by percentile
                closest_bert_token = min(bert_percentiles.items(), key=lambda x: abs(x[1] - maestro_percentile))[0]
                bert_index = bert_tokens.index((closest_bert_token, bert_freq[closest_bert_token]))
                
                mapping_tensor[i, 0] = bert_index
            else:
                mapping_tensor[i, 0] = i  # If we run out of Maestro tokens, keep the original index

    # Create Embedding layer
    embedding = torch.nn.Embedding(256, 1)
    embedding.weight = torch.nn.Parameter(mapping_tensor.float())

    # Save the Embedding layer
    with open(f'../../shift_table/freq_align/{model_name}_align_256_token_mapping.pkl', 'wb') as f:
        torch.save(embedding, f)

    # Create log file
    with open(f'../../shift_table/freq_align/{model_name}_align_token_mapping_log.txt', 'w') as log_file:
        log_file.write(f"Token Mapping Log for {model_name}\n")
        log_file.write(f"Total BERT Tokens: {len(bert_freq)}\n")
        log_file.write(f"Total Maestro Tokens: {len(maestro_freq)}\n\n")
        log_file.write("Mapping for indices 128-255:\n\n")
        log_file.write("Index | Maestro Token | Maestro % | Maestro Count | BERT Token | BERT % | BERT Count\n")
        log_file.write("-" * 90 + "\n")

        for i in range(128, 256):
            maestro_index = i - 128
            if maestro_index < len(maestro_tokens):
                maestro_token, maestro_count = maestro_tokens[maestro_index]
                maestro_percent = maestro_percentiles[maestro_token]
                
                bert_index = mapping_tensor[i, 0].item()
                bert_token, bert_count = bert_tokens[bert_index]
                bert_percent = bert_percentiles[bert_token]

                log_file.write(f"{i:5d} | {maestro_token:13s} | {maestro_percent:8.2%} | {maestro_count:12d} | {bert_token:10s} | {bert_percent:6.2%} | {bert_count:10d}\n")
            else:
                log_file.write(f"{i:5d} | {'N/A':13s} | {'N/A':8s} | {'N/A':12s} | {'N/A':10s} | {'N/A':6s} | {'N/A':10s}\n")

    print(f"Mapping saved to ../../shift_table/freq_align/{model_name}_align_256_token_mapping.pkl")
    print(f"Log file saved to {model_name}_align_token_mapping_log.txt")

def main():
    # Load Maestro frequency data
    maestro_freq = load_json('../maestro_v1_train_token_freq_original.json')

    # Get all model directories
    model_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.endswith('_result')]

    # Process each model
    for model_dir in model_dirs:
        model_name = model_dir.replace('_result', '')
        print(f"Processing {model_name}...")
        process_model(model_name, maestro_freq)

if __name__ == "__main__":
    main()