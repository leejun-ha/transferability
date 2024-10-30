from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import pickle

# Set up logging
logging.basicConfig(filename='token_analysis_all_models.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# models = [
#     "bert-base-uncased",
#     "bert-base-chinese",
#     "bert-base-multilingual-uncased",
#     "bert-base-multilingual-cased",
#     "bert-base-german-cased",
#     "neuralmind/bert-base-portuguese-cased",
#     "tohoku-nlp/bert-base-japanese",
#     "microsoft/codebert-base",
#     "microsoft/codebert-base-mlm",
#     "neulab/codebert-javascript",
#     "neulab/codebert-java",
#     "neulab/codebert-python",
#     "neulab/codebert-c"
# ]
models = [
    "bert-base-uncased",
    "microsoft/codebert-base-mlm",
     "neulab/codebert-javascript"
]

# Dataset mapping
dataset_mapping = {
    "bert-base-uncased": ("wikipedia", "20220301.en"),
    "bert-base-chinese": ("wikipedia", "20220301.zh"),
    "bert-base-multilingual-uncased": ("wikipedia", "20220301.en"),
    "bert-base-multilingual-cased": ("wikipedia", "20220301.en"),
    "bert-base-german-cased": ("wikipedia", "20200501.de"),
    "neuralmind/bert-base-portuguese-cased": ("wikipedia", "20200501.pt"),
    "tohoku-nlp/bert-base-japanese": ("wikipedia", "20200501.ja"),
    "microsoft/codebert-base": ("code_search_net", "all"),
    "microsoft/codebert-base-mlm": ("code_search_net", "all"),
    "neulab/codebert-javascript": ("code_search_net", "javascript"),
    "neulab/codebert-java": ("code_search_net", "java"),
    "neulab/codebert-python": ("code_search_net", "python"),
    "neulab/codebert-c": ("code_search_net", "go")  # Using Go as a proxy for C
}

def load_dataset_for_model(model_name):
    dataset_info = dataset_mapping[model_name]
    if len(dataset_info) == 2:
        return load_dataset(dataset_info[0], dataset_info[1], split='train')
    elif len(dataset_info) == 3:
        return load_dataset(dataset_info[0], dataset_info[1], language=dataset_info[2], split='train')
    else:
        raise ValueError(f"Invalid dataset mapping for {model_name}")

def get_length_bin(length):
    if length < 100:
        return 0
    elif length < 500:
        return 1
    elif length < 1000:
        return 2
    else:
        return 3

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
        token_mapping = {}
        for token, index in vocab.items():
            if token in token_to_rank:
                token_mapping[index] = token_to_rank[token]
            else:
                # For tokens not in the group, map to the end
                token_mapping[index] = 256
        
        mapping_tensor = torch.zeros(256, 1, dtype=torch.long)
        for i in range(256):
            mapping_tensor[i, 0] = token_mapping.get(i, i)
        
        mapping_tensor = mapping_tensor.float()
        mapping_tensor.requires_grad_(True)
        return torch.nn.Embedding.from_pretrained(mapping_tensor, freeze=False)
    
    top_embedding = create_mapping(top_256)
    middle_embedding = create_mapping(middle_256)
    low_embedding = create_mapping(low_256)
    
    model_replace = model_name.replace("/", "_")
    torch.save(top_embedding, f'shift_table/{model_replace}_top_256_token_mapping.pkl')
    torch.save(middle_embedding, f'shift_table/{model_replace}_middle_256_token_mapping.pkl')
    torch.save(low_embedding, f'shift_table/{model_replace}_low_256_token_mapping.pkl')
    
    return top_embedding, middle_embedding, low_embedding

def create_stratified_sample(dataset, sample_ratio=0.01):
    bins = defaultdict(list)
    for i, item in enumerate(dataset):
        # Use 'whole_func_string' which contains both code and documentation
        text = item['whole_func_string']
        length_bin = get_length_bin(len(text))
        bins[length_bin].append(i)
    
    sampled_indices = []
    for bin_indices in bins.values():
        sample_size = max(int(len(bin_indices) * sample_ratio), 1)
        sampled_indices.extend(random.sample(bin_indices, sample_size))
    
    return dataset.select(sampled_indices)

def analyze_tokens(dataset, tokenizer):
    token_counts_per_sequence = []
    total_tokens = 0
    token_frequencies = defaultdict(int)

    for item in tqdm(dataset, desc="Processing Items"):
        # Use 'whole_func_string' which contains both code and documentation
        text = item['whole_func_string']
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        for token in tokens:
            token_frequencies[token] += 1

    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_frequencies, total_tokens

def save_token_counts_in_log(token_counts, filename, max_range=1000, increment=100):
    with open(filename, 'w') as f:
        for i in range(0, max_range + 1, increment):
            in_range_count = sum(1 for count in token_counts if i <= count < i + increment)
            f.write(f"Tokens {i}-{i + increment - 1}: {in_range_count}\n")

def create_visualizations(token_counts_per_sequence, frequencies, model_name):
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    sns.histplot(data=[count for count in token_counts_per_sequence if count <= 1000], 
                 kde=True, bins=100)
    plt.title(f"Distribution of Token Counts per Sequence (0-1000 range)\n{model_name}")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.xlim(0, 1000)

    plt.subplot(2, 2, 2)
    sorted_counts = sorted([count for count in token_counts_per_sequence if count <= 1000])
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative)
    plt.title(f"Cumulative Distribution of Token Counts per Sequence\n{model_name}")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Cumulative Proportion")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)

    plt.subplot(2, 2, 3)
    top_20 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens, freqs = zip(*top_20)
    sns.barplot(x=list(tokens), y=list(freqs))
    plt.title(f"Top 20 Token Frequencies\n{model_name}")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)

    plt.subplot(2, 2, 4)
    sns.boxplot(y=[count for count in token_counts_per_sequence if count <= 1000])
    plt.title(f"Boxplot of Token Counts per Sequence (0-1000 range)\n{model_name}")
    plt.ylabel("Number of Tokens")
    plt.ylim(0, 1000)

    plt.tight_layout()
    plt.savefig(f'token_analysis_{model_name.replace("/", "_")}.png', dpi=300)
    plt.close()

# Main analysis loop
for model_name in models:
    logging.info(f"Analyzing model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset_for_model(model_name)
    # Create sample
    sampled_dataset = create_stratified_sample(dataset)
    
    # Analyze tokens
    average_count, token_counts_per_sequence, token_frequencies, total_tokens = analyze_tokens(sampled_dataset, tokenizer)
    
    # Create and save token mapping shift tables
    top_embedding, middle_embedding, low_embedding = create_token_mapping_shift_tables(tokenizer, token_frequencies, model_name)
    
    # Log information about the created embeddings
    for name, embedding in [("Top 256", top_embedding), ("Middle 256", middle_embedding), ("Low 256", low_embedding)]:
        logging.info(f"{name} Token Mapping Shift Table created for {model_name}")
        logging.info(f"Embedding layer: {embedding}")
        logging.info(f"Embedding weight shape: {embedding.weight.shape}")
        logging.info(f"Number of embeddings: {embedding.num_embeddings}")
        logging.info(f"Embedding dimension: {embedding.embedding_dim}")
    
    # Log results
    logging.info(f"Model: {model_name}")
    logging.info(f"Average Token Count per Sequence: {average_count:.2f}")
    logging.info(f"Total number of tokens: {total_tokens}")
    logging.info(f"Total number of sequences: {len(token_counts_per_sequence)}")
    logging.info("Top 10 Token Frequencies:")
    for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"{token}: {freq}")
    logging.info(f"Original dataset size: {len(dataset)}")
    logging.info(f"Sampled dataset size: {len(sampled_dataset)}")
    
    # Save token counts log
    save_token_counts_in_log(token_counts_per_sequence, f'token_counts_log_{model_name.replace("/", "_")}.txt')
    
    # Create visualizations
    create_visualizations(token_counts_per_sequence, frequencies, model_name)
    
    logging.info(f"Analysis complete for {model_name}")
    logging.info("--------------------")

print("Analysis complete for all models. Results logged to 'token_analysis_all_models.log'.")
print("Visualizations, token count distributions, and token mapping shift tables saved for each model.")

