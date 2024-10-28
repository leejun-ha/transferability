from transformers import BertTokenizer
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load Wikipedia dataset
# Ensure you have `mwparserfromhell` installed for parsing Wikipedia dumps
# For demonstration, we're using a subset; replace with full data path for complete analysis
wikipedia_dataset = load_dataset("wikipedia", "20200501.en", split='train')

# Function to tokenize and analyze token lengths and frequencies
def analyze_tokens(dataset):
    token_lengths = []
    token_frequencies = defaultdict(int)

    for article in tqdm(dataset, desc="Processing Articles"):
        tokens = tokenizer.tokenize(article['text'])
        token_lengths.extend(len(token) for token in tokens)
        for token in tokens:
            token_frequencies[token] += 1

    average_token_length = np.mean(token_lengths)
    return average_token_length, token_frequencies

# Analyze tokens in Wikipedia dataset
average_length, frequencies = analyze_tokens(wikipedia_dataset)

# Print results
print(f"Average Token Length: {average_length}")
print("Sample Token Frequencies:")
for token, freq in list(frequencies.items())[:10]:  # Display top 10 for brevity
    print(f"{token}: {freq}")