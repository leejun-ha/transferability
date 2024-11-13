from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import unidic_lite

import json

from huggingface_hub import snapshot_download

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# List of models and their corresponding datasets
models = [
    # ("bert-base-uncased", "wikimedia/wikipedia", "20231101.en", "text"),
    # ("bert-base-chinese", "wikimedia/wikipedia", "20231101.zh", "text"),
    # ("bert-base-multilingual-uncased", "wikimedia/wikipedia", "20231101.en", "text"),
    # ("bert-base-multilingual-cased", "wikimedia/wikipedia", "20231101.en", "text"),
    # ("bert-base-german-cased", "wikimedia/wikipedia", "20231101.de", "text"),
    # ("neuralmind/bert-base-portuguese-cased", "wikimedia/wikipedia", "20231101.pt", "text"),
    ("tohoku-nlp/bert-base-japanese", "wikimedia/wikipedia", "20231101.ja", "text"),
    # ("microsoft/codebert-base-mlm", "code_search_net", "all", "whole_func_string"),
    # ("neulab/codebert-javascript", "code_search_net", "javascript", "whole_func_string"),
    # ("neulab/codebert-java", "code_search_net", "java", "whole_func_string"),
    # ("neulab/codebert-python", "code_search_net", "python", "whole_func_string"),
    # ("neulab/codebert-c", "code_search_net", "go", "whole_func_string")  # Using 'go' as a substitute for 'c'
]

# Function to create length bins
def get_length_bin(length):
    if length < 100:
        return 0
    elif length < 500:
        return 1
    elif length < 1000:
        return 2
    else:
        return 3

# Create a stratified sample
def create_stratified_sample(dataset, sample_ratio=0.01):
    bins = defaultdict(list)
    for i, article in enumerate(dataset):
        length_bin = get_length_bin(len(article['text'] if 'text' in article else article['whole_func_string']))
        bins[length_bin].append(i)
    
    sampled_indices = []
    for bin_indices in bins.values():
        sample_size = max(int(len(bin_indices) * sample_ratio), 1)
        sampled_indices.extend(random.sample(bin_indices, sample_size))
    
    return dataset.select(sampled_indices)

# Function to tokenize and analyze token counts per sequence
def analyze_tokens(dataset, tokenizer, text_field):
    token_counts_per_sequence = []
    total_tokens = 0
    token_frequencies = defaultdict(int)

    for article in tqdm(dataset, desc="Processing Articles"):
        tokens = tokenizer.tokenize(article[text_field])
        token_count = len(tokens)
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        for token in tokens:
            token_frequencies[token] += 1

    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_frequencies, total_tokens

# Function to save token counts in specified ranges
def save_token_counts_in_ranges(token_counts, filename):
    ranges = [
        (0, 32), (32, 96), (96, 192), (192, 320), (320, 480),
        (480, 672), (672, 896), (896, 1152), (1152, 1440), (1440, 1760)
    ]
    
    with open(filename, 'w') as f:
        for start, end in ranges:
            in_range_count = sum(1 for count in token_counts if start < count <= end)
            f.write(f"{start} < Tokens <= {end}: {in_range_count}\n")
        
        # Add a final range for tokens > 1760
        above_max_count = sum(1 for count in token_counts if count > 1760)
        f.write(f"Tokens > 1760: {above_max_count}\n")

# Function to create visualizations
def create_visualizations(token_counts_per_sequence, frequencies, model_name):
    plt.figure(figsize=(20, 15))

    # Distribution of token counts per sequence in specified ranges
    plt.subplot(2, 2, 1)
    ranges = [
        (0, 32), (32, 96), (96, 192), (192, 320), (320, 480),
        (480, 672), (672, 896), (896, 1152), (1152, 1440), (1440, 1760)
    ]
    range_counts = [sum(1 for count in token_counts_per_sequence if start < count <= end) for start, end in ranges]
    range_labels = [f"{start}-{end}" for start, end in ranges]
    range_labels.append(">1760")
    range_counts.append(sum(1 for count in token_counts_per_sequence if count > 1760))

    plt.bar(range(len(range_counts)), range_counts)
    plt.title("Distribution of Token Counts per Sequence")
    plt.xlabel("Token Count Ranges")
    plt.ylabel("Number of Sequences")
    plt.xticks(range(len(range_labels)), range_labels, rotation=45, ha='right')

    # Cumulative distribution function of token counts
    plt.subplot(2, 2, 2)
    sorted_counts = sorted(token_counts_per_sequence)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative)
    plt.title("Cumulative Distribution of Token Counts per Sequence")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Cumulative Proportion")
    plt.xscale('log')

    # Top 20 token frequencies
    plt.subplot(2, 2, 3)
    top_20 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
    tokens, freqs = zip(*top_20)
    sns.barplot(x=list(tokens), y=list(freqs))
    plt.title("Top 20 Token Frequencies")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)

    # Box plot of token counts
    plt.subplot(2, 2, 4)
    sns.boxplot(y=token_counts_per_sequence)
    plt.title("Boxplot of Token Counts per Sequence")
    plt.ylabel("Number of Tokens")
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f'token_analysis_{model_name}.png', dpi=300)
    plt.close()

def save_token_frequency_ranking(frequencies, filename):
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sorted_frequencies, f, ensure_ascii=False, indent=2)

# Function to tokenize and analyze token counts per sequence
def analyze_tokens(dataset, tokenizer, text_field):
    token_counts_per_sequence = []
    total_tokens = 0
    token_frequencies = defaultdict(int)

    for article in tqdm(dataset, desc="Processing Articles"):
        tokens = tokenizer.tokenize(article[text_field])
        token_count = len(tokens)
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        for token in tokens:
            token_frequencies[token] += 1

    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_frequencies, total_tokens

# Main processing loop
for model_name, dataset_name, dataset_config, text_field in models:
    logging.info(f"Processing model: {model_name}")
    
    model_replace = model_name.replace("/", "_")
    result_dir = model_replace + "_result"
    # Create output directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Download the model files
    try:
        model_path = snapshot_download(repo_id=model_name)
        logging.info(f"Model files downloaded to: {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model {model_name}: {str(e)}")
        continue  # Skip to the next model if download fails
    
    # Initialize the tokenizer using the downloaded files
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False, force_download=True)
        # tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False, force_download=True)
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer for {model_name}: {str(e)}")
        continue  # Skip to the next model if tokenizer initialization fails
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False, force_download=True)
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split='train')
    
    # Create a 1% sample
    sampled_dataset = create_stratified_sample(dataset)
    # Analyze tokens
    average_count, token_counts_per_sequence, frequencies, total_tokens = analyze_tokens(sampled_dataset, tokenizer, text_field)
    
    # Save token counts in ranges
    save_token_counts_in_ranges(token_counts_per_sequence, f'{model_replace}/token_counts_ranges.txt')
    
    # Save token frequency ranking as JSON
    save_token_frequency_ranking(frequencies, f'{model_replace}/token_frequency_ranking.json')
    
    # Log results
    logging.info(f"Average Token Count per Sequence: {average_count:.2f}")
    logging.info(f"Total number of tokens: {total_tokens}")
    logging.info(f"Total number of sequences: {len(token_counts_per_sequence)}")
    logging.info("Top 10 Token Frequencies:")
    for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"{token}: {freq}")
    
    logging.info(f"Original dataset size: {len(dataset)}")
    logging.info(f"Sampled dataset size: {len(sampled_dataset)}")
    
    # Create visualizations
    create_visualizations(token_counts_per_sequence, frequencies, model_replace)
    
    logging.info(f"Analysis complete for {model_replace}. Results saved in the '{model_replace}' directory.")

print("Analysis complete for all models.")