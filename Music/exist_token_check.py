from transformers import AutoTokenizer
import json

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load the frequency ranking JSON
with open('bert-base-uncased_frequency_ranking.json', 'r') as f:
    frequency_ranking = json.load(f)

# Get the tokenizer's vocabulary
vocab = tokenizer.get_vocab()

# Check if all tokens in the frequency ranking are in the tokenizer's vocabulary
all_tokens_present = all(token in vocab for token in frequency_ranking.keys())

# Count how many tokens are present and how many are missing
tokens_present = sum(1 for token in frequency_ranking.keys() if token in vocab)
tokens_missing = sum(1 for token in frequency_ranking.keys() if token not in vocab)

print(f"All tokens present in tokenizer vocabulary: {all_tokens_present}")
print(f"Tokens present: {tokens_present}")
print(f"Tokens missing: {tokens_missing}")

# If there are missing tokens, print them
if tokens_missing > 0:
    missing_tokens = [token for token in frequency_ranking.keys() if token not in vocab]
    print("Missing tokens:")
    print(missing_tokens)