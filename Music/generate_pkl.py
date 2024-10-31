import torch
import json
from transformers import AutoTokenizer
import torch.nn as nn

def create_token_mapping_shift_tables(tokenizer, frequencies, model_name):
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Sort tokens by frequency
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # Create three groups of 256 tokens each
    top_256 = sorted_tokens[:256]
    middle_256 = sorted_tokens[len(sorted_tokens)//2 - 128:len(sorted_tokens)//2 + 128]
    low_256 = sorted_tokens[-256:]
    
    def create_mapping(token_group):
        token_to_rank = {token: rank for rank, (token, _) in enumerate(token_group)}
        mapping_tensor = torch.full((256, 1), 256, dtype=torch.long)
        
        for token, _ in token_group:
            if token in vocab:
                index = vocab[token]
                rank = token_to_rank[token]
                if rank < 256:
                    mapping_tensor[rank, 0] = index
        
        return nn.Embedding.from_pretrained(mapping_tensor.float(), freeze=False)
    
    top_embedding = create_mapping(top_256)
    middle_embedding = create_mapping(middle_256)
    low_embedding = create_mapping(low_256)
    
    model_replace = model_name.replace("/", "_")
    torch.save(top_embedding, f'shift_table/{model_replace}_top_256_token_mapping.pkl')
    torch.save(middle_embedding, f'shift_table/{model_replace}_middle_256_token_mapping.pkl')
    torch.save(low_embedding, f'shift_table/{model_replace}_low_256_token_mapping.pkl')
    
    return top_embedding, middle_embedding, low_embedding

# Usage
models = [
    "neulab/codebert-javascript",
    "neulab/codebert-java",
    "neulab/codebert-python",
    "neulab/codebert-c"
]

for model_name in models:
    print(f"Creating mapping pkl files for {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load frequency ranking
    with open(f'{model_name.replace("/", "_")}_frequency_ranking.json', 'r') as f:
        frequencies = json.load(f)
    
    # Create and save token mapping shift tables
    top_embedding, middle_embedding, low_embedding = create_token_mapping_shift_tables(tokenizer, frequencies, model_name)
    
    print(f"Mapping pkl files created for {model_name}")
    print(f"Top embedding shape: {top_embedding.weight.shape}")
    print(f"Middle embedding shape: {middle_embedding.weight.shape}")
    print(f"Low embedding shape: {low_embedding.weight.shape}")
    print("--------------------")

print("Mapping pkl files created for all models.")

def inspect_embedding(file_path, tokenizer, token_group):
    embedding = torch.load(file_path)
    print(f"Embedding shape: {embedding.weight.shape}")
    print(f"Type of loaded data: {type(embedding)}")
    print("\nSample mappings:")
    for i, (token, _) in enumerate(token_group[:10]):  # Show first 10 mappings
        token_id = tokenizer.convert_tokens_to_ids(token)
        mapped_value = embedding.weight[i, 0].item()
        print(f"Rank: {i}, Token: {token}, Tokenizer ID: {token_id}, Mapped value: {mapped_value}")

# After creating the mappings, inspect them
for model_name in models:
    model_replace = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f'{model_replace}_frequency_ranking.json', 'r') as f:
        frequencies = json.load(f)
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nInspecting embeddings for {model_name}")
    for embedding_type, token_group in [('top', sorted_tokens[:256]), 
                                        ('middle', sorted_tokens[len(sorted_tokens)//2 - 128:len(sorted_tokens)//2 + 128]), 
                                        ('low', sorted_tokens[-256:])]:
        file_path = f'shift_table/{model_replace}_{embedding_type}_256_token_mapping.pkl'
        print(f"\n{embedding_type.capitalize()} 256 Token Mapping:")
        inspect_embedding(file_path, tokenizer, token_group)