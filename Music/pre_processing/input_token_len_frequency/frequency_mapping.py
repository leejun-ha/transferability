import torch
import json
from transformers import AutoTokenizer
import torch.nn as nn
import os

def create_token_mapping_shift_tables(tokenizer, frequencies, model_name, top_count, middle_count, low_count):
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Sort tokens by frequency
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # Create three groups of tokens
    top_tokens = sorted_tokens[:top_count]
    middle_start = len(sorted_tokens)//2 - middle_count//2
    middle_tokens = sorted_tokens[middle_start:middle_start + middle_count]
    low_tokens = sorted_tokens[-low_count:]
    
    def create_mapping(token_group, count, is_top_or_low=False):
        token_to_rank = {token: rank for rank, (token, _) in enumerate(token_group)}
        mapping_tensor = torch.full((256, 1), 256, dtype=torch.long)
        
        for token, _ in token_group:
            if token in vocab:
                index = vocab[token]
                rank = token_to_rank[token]
                if rank < 256:
                    if is_top_or_low:
                        if rank < 128:
                            mapping_tensor[rank + 128, 0] = index
                        else:
                            mapping_tensor[rank - 128, 0] = index
                    else:
                        mapping_tensor[rank, 0] = index
        
        return nn.Embedding.from_pretrained(mapping_tensor.float(), freeze=False)
    
    top_embedding = create_mapping(top_tokens, top_count, is_top_or_low=True)
    middle_embedding = create_mapping(middle_tokens, middle_count)
    low_embedding = create_mapping(low_tokens, low_count, is_top_or_low=True)
    
    model_replace = model_name.replace("/", "_")
    os.makedirs('../shift_table', exist_ok=True)
    torch.save(top_embedding, f'../shift_table/{model_replace}_top_256_token_mapping.pkl')
    torch.save(middle_embedding, f'../shift_table/{model_replace}_middle_256_token_mapping.pkl')
    torch.save(low_embedding, f'../shift_table/{model_replace}_low_256_token_mapping.pkl')
    
    return top_embedding, middle_embedding, low_embedding

def inspect_embedding(file_path, tokenizer, token_group, is_top_or_low=False):
    embedding = torch.load(file_path)
    print(f"Embedding shape: {embedding.weight.shape}")
    print(f"Type of loaded data: {type(embedding)}")
    print("\nSample mappings:")
    for i, (token, _) in enumerate(token_group[:20]):  # Show first 20 mappings
        token_id = tokenizer.convert_tokens_to_ids(token)
        if is_top_or_low:
            if i < 128:
                mapped_value = embedding.weight[i + 128, 0].item()
            else:
                mapped_value = embedding.weight[i - 128, 0].item()
        else:
            mapped_value = embedding.weight[i, 0].item()
        print(f"Rank: {i}, Token: {token}, Tokenizer ID: {token_id}, Mapped value: {mapped_value}")

# Usage
models = [
    "bert-base-uncased",
    "bert-base-chinese",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-german-cased",
    "neuralmind/bert-base-portuguese-cased",
    "tohoku-nlp/bert-base-japanese",
    "microsoft/codebert-base-mlm",
    "neulab/codebert-javascript",
    "neulab/codebert-java",
    "neulab/codebert-python",
    "neulab/codebert-c"
]

# Fixed counts for top, middle, and low
top_count = 256
middle_count = 256
low_count = 256

for model_name in models:
    print(f"Creating mapping pkl files for {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load frequency ranking
    model_replace = model_name.replace("/", "_")
    with open(f'{model_replace}/token_frequency_ranking.json', 'r') as f:
        frequencies = json.load(f)
    
    # Create and save token mapping shift tables
    top_embedding, middle_embedding, low_embedding = create_token_mapping_shift_tables(
        tokenizer, frequencies, model_name, top_count, middle_count, low_count)
    
    print(f"Mapping pkl files created for {model_name}")
    print(f"Top embedding shape: {top_embedding.weight.shape}")
    print(f"Middle embedding shape: {middle_embedding.weight.shape}")
    print(f"Low embedding shape: {low_embedding.weight.shape}")
    print("--------------------")

print("Mapping pkl files created for all models.")

# After creating the mappings, inspect them
for model_name in models:
    model_replace = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f'{model_replace}/token_frequency_ranking.json', 'r') as f:
        frequencies = json.load(f)
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nInspecting embeddings for {model_name}")
    for embedding_type, token_group, is_top_or_low in [
        ('top', sorted_tokens[:256], True), 
        ('middle', sorted_tokens[len(sorted_tokens)//2 - 128:len(sorted_tokens)//2 + 128], False), 
        ('low', sorted_tokens[-256:], True)
    ]:
        file_path = f'shift_table/{model_replace}_{embedding_type}_256_token_mapping.pkl'
        print(f"\n{embedding_type.capitalize()} 256 Token Mapping:")
        inspect_embedding(file_path, tokenizer, token_group, is_top_or_low)