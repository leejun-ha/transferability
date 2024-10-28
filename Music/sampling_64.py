from transformers import BertTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(filename='token_analysis_64.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load Wikipedia dataset
wikipedia_dataset = load_dataset("wikipedia", "20200501.en", split='train')

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
        length_bin = get_length_bin(len(article['text']))
        bins[length_bin].append(i)
    
    sampled_indices = []
    for bin_indices in bins.values():
        sample_size = max(int(len(bin_indices) * sample_ratio), 1)
        sampled_indices.extend(random.sample(bin_indices, sample_size))
    
    return dataset.select(sampled_indices)

# Create a 1% sample
sampled_dataset = create_stratified_sample(wikipedia_dataset)

# Function to tokenize and analyze token counts per sequence
def analyze_tokens(dataset):
    token_counts_per_sequence = []
    total_tokens = 0
    token_frequencies = defaultdict(int)

    for article in tqdm(dataset, desc="Processing Articles"):
        tokens = tokenizer.tokenize(article['text'])
        token_count = len(tokens)
        token_counts_per_sequence.append(token_count)
        total_tokens += token_count
        for token in tokens:
            token_frequencies[token] += 1

    average_token_count = np.mean(token_counts_per_sequence)
    return average_token_count, token_counts_per_sequence, token_frequencies, total_tokens

# Analyze tokens in sampled Wikipedia dataset
average_count, token_counts_per_sequence, frequencies, total_tokens = analyze_tokens(sampled_dataset)

# Function to save token counts in increments of 100
def save_token_counts_in_log(token_counts, max_range=1000, increment=100):
    with open('token_counts_log_64.txt', 'w') as f:
        for i in range(0, max_range + 1, increment):
            in_range_count = sum(1 for count in token_counts if i <= count < i + increment)
            f.write(f"Tokens {i}-{i + increment - 1}: {in_range_count}\n")

# Function to save token counts for specific sequence lengths
def save_specific_token_counts(token_counts, specific_lengths):
    with open('specific_token_counts_log_64.txt', 'w') as f:
        for length in specific_lengths:
            exact_count = sum(1 for count in token_counts if count == length)
            less_than_count = sum(1 for count in token_counts if count <= length)
            f.write(f"Tokens = {length}: {exact_count}\n")
            f.write(f"Tokens <= {length}: {less_than_count}\n\n")

# Save token counts logs
save_token_counts_in_log(token_counts_per_sequence)
specific_lengths = [0, 64, 128, 256, 384, 512, 768, 1024, 2048]
save_specific_token_counts(token_counts_per_sequence, specific_lengths)

# Log results
logging.info(f"Average Token Count per Sequence: {average_count:.2f}")
logging.info(f"Total number of tokens: {total_tokens}")
logging.info(f"Total number of sequences: {len(token_counts_per_sequence)}")
logging.info("Top 10 Token Frequencies:")
for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
    logging.info(f"{token}: {freq}")

logging.info(f"Original dataset size: {len(wikipedia_dataset)}")
logging.info(f"Sampled dataset size: {len(sampled_dataset)}")

# Visualizations
plt.figure(figsize=(20, 20))

# Fine-grained distribution of token counts per sequence (0-1000 range)
plt.subplot(3, 2, 1)
sns.histplot(data=[count for count in token_counts_per_sequence if count <= 1000], 
             kde=True, bins=100)
plt.title("Distribution of Token Counts per Sequence (0-1000 range)")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.xlim(0, 1000)

# Cumulative distribution function of token counts
plt.subplot(3, 2, 2)
sorted_counts = sorted([count for count in token_counts_per_sequence if count <= 1000])
cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
plt.plot(sorted_counts, cumulative)
plt.title("Cumulative Distribution of Token Counts per Sequence")
plt.xlabel("Number of Tokens")
plt.ylabel("Cumulative Proportion")
plt.xlim(0, 1000)
plt.ylim(0, 1)

# Top 20 token frequencies
plt.subplot(3, 2, 3)
top_20 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
tokens, freqs = zip(*top_20)
sns.barplot(x=list(tokens), y=list(freqs))
plt.title("Top 20 Token Frequencies")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.xticks(rotation=90)

# Box plot of token counts (0-1000 range)
plt.subplot(3, 2, 4)
sns.boxplot(y=[count for count in token_counts_per_sequence if count <= 1000])
plt.title("Boxplot of Token Counts per Sequence (0-1000 range)")
plt.ylabel("Number of Tokens")
plt.ylim(0, 1000)

# Specific sequence length distribution
plt.subplot(3, 2, (5, 6))
specific_counts = [sum(1 for count in token_counts_per_sequence if count <= length) for length in specific_lengths]
plt.bar(range(len(specific_lengths)), specific_counts, tick_label=specific_lengths)
plt.title("Cumulative Distribution for Specific Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")
plt.xticks(rotation=45)

for i, count in enumerate(specific_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('token_analysis_64.png', dpi=300)
plt.close()

print("Analysis complete. Results logged to 'token_analysis_64.log' and visualizations saved to 'token_analysis.png'.")
print("Token count distribution saved to 'token_counts_log.txt' and 'specific_token_counts_log.txt'.")