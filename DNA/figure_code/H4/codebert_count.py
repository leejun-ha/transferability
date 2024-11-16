from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import os
from huggingface_hub import snapshot_download

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Model and dataset configurations
model_name = "microsoft/codebert-base-mlm"
datasets_config = [
    ("cc_news", None, "text"),
    ("openwebtext", None, "text"),
    ("wikipedia", "20231101.en", "text"),
    ("code_search_net", "all", "whole_func_string")
]

def save_token_counts(token_counts, filename, truncate=True):
    max_length = 512 if truncate else max(token_counts)
    counts = defaultdict(int)
    
    for count in token_counts:
        if truncate:
            counts[min(count, 512)] += 1
        else:
            counts[count] += 1
    
    with open(filename, 'w') as f:
        for length in range(1, max_length + 1):
            f.write(f"{length}: {counts[length]}\n")

def create_sample(dataset, sample_ratio=0.01):
    sample_size = max(int(len(dataset) * sample_ratio), 1)
    return dataset.select(random.sample(range(len(dataset)), sample_size))

def analyze_tokens(dataset, tokenizer, text_field):
    token_counts_per_sequence = []
    token_counts_per_sequence_no_truncation = []
    total_tokens = 0
    token_frequencies = defaultdict(int)
    
    for item in tqdm(dataset, desc="Processing Items"):
        text = item[text_field]
        if isinstance(text, list):  # Handle cases where text field is a list
            text = " ".join(text)
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        token_counts_per_sequence_no_truncation.append(token_count)
        
        if token_count > 512:
            tokens = tokens[:512]
            token_count = 512
        
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        
        for token in tokens:
            token_frequencies[token] += 1
    
    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_counts_per_sequence_no_truncation, token_frequencies, total_tokens

def create_visualizations(token_counts_per_sequence, frequencies, model_name, dataset_name, result_dir):
    plt.figure(figsize=(15, 10))
    min_length = min(token_counts_per_sequence)
    max_length = max(token_counts_per_sequence)
    bins = max_length - min_length + 1  # One bin for each possible token length
    
    counts, edges, _ = plt.hist(token_counts_per_sequence, bins=bins, range=(min_length, max_length), density=False)
    plt.clf()  # Clear the current figure
    
    min_count = min(counts[counts > 0])  # Find the minimum non-zero count
    max_count = max(counts)
    
    plt.hist(token_counts_per_sequence, bins=bins, range=(min_length, max_length), density=False)
    plt.yscale('log')  # Use log scale for y-axis
    plt.ylim(min_count, max_count)  # Set y-axis range
    
    plt.title(f"Distribution of Input Token Lengths - {model_name} - {dataset_name}")
    plt.xlabel("Input Token Length")
    plt.ylabel("Count (log scale)")
    plt.xlim(min_length, max_length)
    
    # Add text annotations for min and max counts
    plt.text(0.02, 0.98, f"Min count: {min_count}", transform=plt.gca().transAxes, va='top', ha='left')
    plt.text(0.02, 0.93, f"Max count: {max_count}", transform=plt.gca().transAxes, va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/{dataset_name}_input_token_length_distribution.png', dpi=300)
    plt.close()

# Main processing
logging.info(f"Processing model: {model_name}")
model_replace = model_name.replace("/", "_")
result_dir = model_replace + "_result"
os.makedirs(result_dir, exist_ok=True)

try:
    model_path = snapshot_download(repo_id=model_name)
    logging.info(f"Model files downloaded to: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False, force_download=True)
except Exception as e:
    logging.error(f"Failed to download model or initialize tokenizer for {model_name}: {str(e)}")
    exit(1)

for dataset_name, dataset_config, text_field in datasets_config:
    logging.info(f"Processing dataset: {dataset_name}")
    
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split='train')
        else:
            dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        continue
    
    sampled_dataset = create_sample(dataset, sample_ratio=0.01)  # 1% sample

    average_count, token_counts_per_sequence, token_counts_per_sequence_no_truncation, frequencies, total_tokens = analyze_tokens(sampled_dataset, tokenizer, text_field)

    save_token_counts(token_counts_per_sequence, f'{result_dir}/{dataset_name}_token_counts_truncated.txt', truncate=True)
    save_token_counts(token_counts_per_sequence_no_truncation, f'{result_dir}/{dataset_name}_token_counts_original.txt', truncate=False)

    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Average Token Count per Sequence: {average_count:.2f}")
    logging.info(f"Total number of tokens: {total_tokens}")
    logging.info(f"Total number of sequences: {len(token_counts_per_sequence)}")
    logging.info("Top 10 Token Frequencies:")
    for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"{token}: {freq}")
    logging.info(f"Original dataset size: {len(dataset)}")
    logging.info(f"Sampled dataset size: {len(sampled_dataset)}")

    create_visualizations(token_counts_per_sequence, frequencies, model_replace, dataset_name, result_dir)
    logging.info(f"Analysis complete for {dataset_name}. Results saved in the '{result_dir}' directory.")

print("Analysis complete for all datasets.")