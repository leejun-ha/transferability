import torch
import pickle
import transformers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = '')
args = vars(parser.parse_args())

model_name = args["model"]
model_replace = model_name.replace("/", "_")
print(model_replace)
# Get the BERT tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Get the vocabulary
vocab = tokenizer.get_vocab()

# Create the mapping
unused_token_count = 0
token_mapping = {}
for token, index in vocab.items():
    if token.startswith('[unused'):
        unused_token_count += 1
    else:
        # Map non-unused tokens to your non-textual data tokens
        token_mapping[index] = unused_token_count + index

# Create a tensor from the mapping with size [256, 1]
mapping_tensor = torch.zeros(256, 1, dtype=torch.long)
for i in range(min(256, len(vocab))):
    mapping_tensor[i, 0] = token_mapping.get(i, i)

mapping_tensor = mapping_tensor.float()
mapping_tensor.requires_grad_(True)
# Create an Embedding layer
embedding_layer = torch.nn.Embedding.from_pretrained(mapping_tensor, freeze=False)

# Save the embedding layer as a pkl file
torch.save(embedding_layer, f'shift_table/{model_replace}_bert_token_mapping.pkl')

# Print information about the created embedding
print("Embedding layer:", embedding_layer)
print("Embedding weight shape:", embedding_layer.weight.shape)
print("Number of embeddings:", embedding_layer.num_embeddings)
print("Embedding dimension:", embedding_layer.embedding_dim)
print("Embedding weights:")
print(embedding_layer.weight)
print("First embedding vector:")
print(embedding_layer.weight[0])
print("Padding index:", embedding_layer.padding_idx)
print("Embeddings frozen:", embedding_layer.weight.requires_grad == False)