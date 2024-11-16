import torch
import json
from transformers import AutoTokenizer
import torch.nn as nn
import os
import numpy as np
from huggingface_hub import snapshot_download

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

def create_low_token_mapping(tokenizer, frequencies, model_name, low_count=256):
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # Select tokens from the bottom 33%
    bottom_third = sorted_tokens[int(len(sorted_tokens) * 2/3):]
    
    # Divide the bottom third into 128 zones
    zone_size = len(bottom_third) // 128
    low_tokens = []
    
    for i in range(128):
        start_idx = i * zone_size
        end_idx = (i + 1) * zone_size if i < 127 else len(bottom_third)
        zone_tokens = bottom_third[start_idx:end_idx]
        if zone_tokens:
            # Select the token with frequency closest to the mean of the zone
            zone_freq = [freq for _, freq in zone_tokens]
            zone_mean = np.mean(zone_freq)
            selected_token = min(zone_tokens, key=lambda x: abs(x[1] - zone_mean))
            low_tokens.append(selected_token)
    
    # If we have less than 256 tokens, repeat the process to fill the remaining slots
    while len(low_tokens) < 256:
        for token in low_tokens[:]:
            if len(low_tokens) < 256:
                low_tokens.append(token)
            else:
                break
    
    mapping_tensor = torch.full((256, 1), 256, dtype=torch.long)
    
    for rank, (token, _) in enumerate(low_tokens):
        if token in vocab:
            index = vocab[token]
            if rank < 128:
                mapping_tensor[rank + 128, 0] = index
            else:
                mapping_tensor[rank - 128, 0] = index
    
    low_embedding = nn.Embedding.from_pretrained(mapping_tensor.float(), freeze=False)
    
    model_replace = model_name.replace("/", "_")
    os.makedirs('../../shift_table/avg', exist_ok=True)
    torch.save(low_embedding, f'../../shift_table/avg/{model_replace}_low_256_token_mapping.pkl')
    
    # Save token names and frequencies to a log file
    log_file_path = f'../../shift_table/avg/{model_replace}_low_256_token_mapping_log.txt'
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Low token mapping for {model_name}\n")
        log_file.write("Rank\tToken\tFrequency\n")
        for rank, (token, freq) in enumerate(low_tokens):
            log_file.write(f"{rank}\t{token}\t{freq}\n")
    
    return low_embedding, low_tokens

def inspect_low_embedding(file_path, tokenizer, low_tokens):
    embedding = torch.load(file_path)
    print(f"Embedding shape: {embedding.weight.shape}")
    print(f"Type of loaded data: {type(embedding)}")
    print("\nSample mappings:")
    for i, (token, freq) in enumerate(low_tokens[:20]):  # Show first 20 mappings
        token_id = tokenizer.convert_tokens_to_ids(token)
        if i < 128:
            mapped_value = embedding.weight[i + 128, 0].item()
        else:
            mapped_value = embedding.weight[i - 128, 0].item()
        print(f"Rank: {i}, Token: {token}, Frequency: {freq}, Tokenizer ID: {token_id}, Mapped value: {mapped_value}")

# Models to process
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

for model_name in models:
    print(f"Creating low token mapping for {model_name}")
    
    try:
        model_path = snapshot_download(repo_id=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False, force_download=True)
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        continue
    
    model_replace = model_name.replace("/", "_")
    result_dir = model_replace + "_result"
    with open(f'{result_dir}/token_frequency_ranking.json', 'r') as f:
        frequencies = json.load(f)
    
    low_embedding, low_tokens = create_low_token_mapping(tokenizer, frequencies, model_name)
    
    print(f"Low embedding shape: {low_embedding.weight.shape}")
    print("--------------------")

    # Inspect the created low token mapping
    file_path = f'../../shift_table/avg/{model_replace}_low_256_token_mapping.pkl'
    print(f"\nInspecting low token mapping for {model_name}")
    inspect_low_embedding(file_path, tokenizer, low_tokens)
    print("--------------------")

print("Low token mapping created and inspected for all models.")