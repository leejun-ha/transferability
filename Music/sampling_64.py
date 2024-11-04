from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(filename='token_analysis_range.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Initialize the BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
# Load Wikipedia dataset
# wikipedia_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train') 
wikipedia_dataset = load_dataset("wikimedia/wikipedia", "20231101.de", split='train') 
# wikipedia_dataset = load_dataset("code_search_net", "all", split='train') 

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
        # length_bin = get_length_bin(len(article['whole_func_string']))
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
        # tokens = tokenizer.tokenize(article['whole_func_string'])
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

# Function to save token counts in specified ranges
def save_token_counts_in_ranges(token_counts):
    ranges = [
        (0, 32), (32, 96), (96, 192), (192, 320), (320, 480),
        (480, 672), (672, 896), (896, 1152), (1152, 1440), (1440, 1760)
    ]
    
    with open('token_counts_ranges_64_2.txt', 'w') as f:
        for start, end in ranges:
            in_range_count = sum(1 for count in token_counts if start < count <= end)
            f.write(f"{start} < Tokens <= {end}: {in_range_count}\n")
        
        # Add a final range for tokens > 1760
        above_max_count = sum(1 for count in token_counts if count > 1760)
        f.write(f"Tokens > 1760: {above_max_count}\n")

# Save token counts in ranges
save_token_counts_in_ranges(token_counts_per_sequence)

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
plt.xscale('log')  # Use log scale for x-axis to better show the distribution

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
plt.yscale('log')  # Use log scale for y-axis to better show the distribution

plt.tight_layout()
plt.savefig('token_analysis_range_2.png', dpi=300)
plt.close()

print("Analysis complete. Results logged to 'token_analysis.log' and visualizations saved to 'token_analysis.png'.")
print("Token count distribution saved to 'token_counts_ranges.txt'.")