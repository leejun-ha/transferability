from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# List of models and their corresponding datasets
models = [
    ("bert-base-multilingual-uncased", "wikimedia/wikipedia"),
    ("bert-base-multilingual-cased", "wikimedia/wikipedia"),
]

# Languages to include
languages = ['en', 'es', 'fr', 'de']

def get_length_bin(length):
    if length <= 512:
        bins = [32, 64, 128, 256, 384, 512]
        return min(bins, key=lambda x: abs(x - length))
    else:
        truncated_length = length - 512
        bins = [32, 64, 128, 256, 384, 512]
        return min(bins, key=lambda x: abs(x - truncated_length))

def create_stratified_sample(dataset, sample_ratio=0.01):
    sampled_indices = random.sample(range(len(dataset)), int(len(dataset) * sample_ratio))
    return dataset.select(sampled_indices)

def analyze_tokens(dataset, tokenizer):
    token_counts_per_sequence = []
    total_tokens = 0
    token_frequencies = defaultdict(int)
    for article in tqdm(dataset, desc="Processing Articles"):
        tokens = tokenizer.tokenize(article['text'])
        token_count = len(tokens)
        
        if token_count > 512:
            tokens = tokens[:512]
            token_count = 512
        
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        for token in tokens:
            token_frequencies[token] += 1
    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_frequencies, total_tokens

def save_token_counts_in_ranges(token_counts, filename):
    bins = [32, 64, 128, 256, 384, 512]
    bin_counts = {bin: 0 for bin in bins}
    
    for count in token_counts:
        nearest_bin = get_length_bin(count)
        bin_counts[nearest_bin] += 1
    
    with open(filename, 'w') as f:
        for bin in bins:
            f.write(f"{bin}: {bin_counts[bin]}\n")

def save_token_frequency_ranking(frequencies, filename):
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sorted_frequencies, f, ensure_ascii=False, indent=2)

def create_visualizations(token_counts_per_sequence, frequencies, model_name):
    plt.figure(figsize=(20, 15))

    # Distribution of token counts per sequence in specified ranges
    plt.subplot(2, 2, 1)
    bins = [32, 64, 128, 256, 384, 512]
    bin_counts = {bin: 0 for bin in bins}
    
    for count in token_counts_per_sequence:
        nearest_bin = get_length_bin(count)
        bin_counts[nearest_bin] += 1
    
    plt.bar(range(len(bins)), [bin_counts[bin] for bin in bins])
    plt.title("Distribution of Token Counts per Sequence")
    plt.xlabel("Token Count Ranges")
    plt.ylabel("Number of Sequences")
    plt.xticks(range(len(bins)), bins, rotation=45, ha='right')

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

# Main processing loop
for model_name, dataset_name in models:
    logging.info(f"Processing model: {model_name}")
    model_replace = model_name.replace("/", "_")
    result_dir = model_replace + "_result"
    os.makedirs(result_dir, exist_ok=True)

    try:
        model_path = snapshot_download(repo_id=model_name)
        logging.info(f"Model files downloaded to: {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model {model_name}: {str(e)}")
        continue

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer for {model_name}: {str(e)}")
        continue

    # Load and combine datasets for all languages
    datasets = []
    for lang in languages:
        dataset = load_dataset(dataset_name, f"20231101.{lang}", split='train')
        datasets.append(dataset)
    
    combined_dataset = concatenate_datasets(datasets)
    sampled_dataset = create_stratified_sample(combined_dataset)

    average_count, token_counts_per_sequence, frequencies, total_tokens = analyze_tokens(sampled_dataset, tokenizer)

    save_token_counts_in_ranges(token_counts_per_sequence, f'{result_dir}/token_counts_ranges.txt')
    save_token_frequency_ranking(frequencies, f'{result_dir}/token_frequency_ranking.json')

    logging.info(f"Average Token Count per Sequence: {average_count:.2f}")
    logging.info(f"Total number of tokens: {total_tokens}")
    logging.info(f"Total number of sequences: {len(token_counts_per_sequence)}")
    logging.info("Top 10 Token Frequencies:")
    for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"{token}: {freq}")
    logging.info(f"Original combined dataset size: {len(combined_dataset)}")
    logging.info(f"Sampled dataset size: {len(sampled_dataset)}")

    create_visualizations(token_counts_per_sequence, frequencies, model_replace)
    logging.info(f"Analysis complete for {model_replace}. Results saved in the '{result_dir}' directory.")

print("Analysis complete for both multilingual models.")