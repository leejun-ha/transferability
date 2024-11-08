import os
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np

import argparse
import tensorboardX

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time

from torch.utils.data import TensorDataset, DataLoader, Subset

# Function to save accuracy results to a text file
def save_accuracy_log(file_path, combined_acc, total_samples, individual_results):
    with open(file_path, 'w') as f:
        f.write(f'Evaluation results:\n\n')
        f.write(f'Combined dataset results:\n')
        f.write(f'Total samples: {total_samples}\n')
        f.write(f'Combined accuracy: {combined_acc:.4f}\n\n')
        f.write(f'Individual test_token_len results:\n')
        for test_token_len, (correct, total) in individual_results.items():
            acc = correct / total
            f.write(f'test_token_len {test_token_len}:\n')
            f.write(f'  Samples: {total}\n')
            f.write(f'  Correct: {correct}\n')
            f.write(f'  Accuracy: {acc:.4f}\n\n')

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--model', type = str, default = 'bert-base-uncased')
parser.add_argument('--seed', type = int, default = 2020)
parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'])
parser.add_argument('--split', type = str, choices=['dev', 'test'], default='test')
parser.add_argument('--state_dict', type = str, default = '')
parser.add_argument('--batch_size', '-b', type = int, default = 16)
parser.add_argument('--shift_table', type = str, default = '')
parser.add_argument('--step', type = str)

parser.add_argument('--logdir', type = str, default = '/home/junha/transferability/Music/pre_processing/input_token_len_frequency/acc')
parser.add_argument('--datadir', type = str)

parser.add_argument('--token_len', type = int, default = 128)
# parser.add_argument('--test_token_len', type = int, default = 128)
parser.add_argument('--ranking', type = str, default = None)
parser.add_argument('--oov', type = int, default = 0)
# Add a new argument for sampling seed
parser.add_argument('--sampling_seed', type=int, default=42, help='Seed for sampling data')

args = vars(parser.parse_args())

# Set the sampling seed
sampling_seed = args['sampling_seed']
#if args['filename'] == None:
#    args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}'
print(args)
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

if args['step'] is not None:
    if args['step'].isdigit():
        args['step'] = int(args['step'])
    else:
        # Keep it as a string if it cannot be converted to an integer
        pass

torch.backends.cudnn.benchmark = True

## load a dev dataset
model_name = args["model"]
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
batch_size = args['batch_size'] #2 devices

model_replace = args["model"]
model_replace = model_replace.replace("/", "_")
print(model_replace)

ranking = args["ranking"]

if(args["ranking"] == None):
    if(args["oov"] == 2):
        args["shift_table"] = ""
    else:
        args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_bert_token_mapping.pkl')
else:
    args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_' + ranking + '_256_token_mapping.pkl')

token_len = args["token_len"]    
data_path = os.path.join(args['datadir'], model_replace)

# List of test token lengths
test_token_lengths = [64, 128, 256, 384, 512]

# Modified load_data function to return the full dataset
def load_data(data_path, model_replace, split, test_token_len):
    data_file = os.path.join(data_path, f'{model_replace}_{test_token_len}_{split}_data.pkl')
    label_file = os.path.join(data_path, f'{model_replace}_{test_token_len}_{split}_label.pkl')
    
    data = torch.load(data_file)
    label = torch.load(label_file)
    
    return TensorDataset(data, label)

composer2id = torch.load(os.path.join(data_path, 'composer2id_map.pkl'))    
num_labels = len(composer2id)

config = transformers.AutoConfig.from_pretrained(model_name, num_labels = num_labels)
model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
if args['shift_table'] != '':
    if args['ranking'] == None :
        if( args['oov'] == 1):
            state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_pretrain_seed{args["seed"]}_tokenlen{token_len}_filtered_table__{args["step"]}.pkl')
        else:    
            state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_pretrain_seed{args["seed"]}_tokenlen{token_len}_table__{args["step"]}.pkl')
    else:      
        state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_pretrain_seed{args["seed"]}_{args["ranking"]}_tokenlen{token_len}_table__{args["step"]}.pkl')
else:
    state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}_{args["step"]}.pkl')
model.load_state_dict(torch.load(state_dict_path), strict=False)
model.cuda()
#model = torch.nn.DataParallel(model)
if args['shift_table'] != '':
    shift_table = torch.load(args['shift_table']).cuda()

# writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], 
#                                     filename_suffix=f'_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}')

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    total_acc = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            if args['shift_table'] != '':
                input_ids = shift_table(input_ids).long().squeeze()
            labels = labels.to(device)

            logits = model(input_ids=input_ids)
            ans = torch.mode(torch.argmax(logits[0], dim=-1)).values
            total_acc += torch.sum(torch.eq(ans, labels)).item()
            total_samples += labels.size(0)
    
    return total_acc, total_samples

def sample_indices(dataset_size, num_samples, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_size, size=num_samples, replace=False)

# Load all datasets
datasets = {}
for test_token_len in test_token_lengths:
    dataset = load_data(data_path, model_replace, args['split'], test_token_len)
    datasets[test_token_len] = dataset

# Get the size of the 512 token_len dataset
sample_size = len(datasets[512])
print(f"Sampling {sample_size} data points from each dataset based on the 512 token_len dataset size.")

# Sample equal number of data points from each dataset
sampled_datasets = {}
for test_token_len, dataset in datasets.items():
    if len(dataset) < sample_size:
        print(f"Warning: Dataset for token_len {test_token_len} has fewer than {sample_size} samples. Using all available data.")
        sampled_datasets[test_token_len] = dataset
    else:
        indices = sample_indices(len(dataset), sample_size, sampling_seed)
        sampled_datasets[test_token_len] = Subset(dataset, indices)

# Evaluate on individual test_token_len datasets
individual_results = {}
for test_token_len, dataset in sampled_datasets.items():
    print(f"Evaluating on test_token_len {test_token_len}...")
    iterator = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True)
    correct, total = evaluate(model, iterator)
    individual_results[test_token_len] = (correct, total)
    print(f'Accuracy for test_token_len {test_token_len}: {correct/total:.4f}')

# Calculate combined accuracy
correct_count_sum = sum(correct for correct, _ in individual_results.values())
total_count_sum = sum(total for _, total in individual_results.values())
combined_accuracy = correct_count_sum / total_count_sum

print(f'Combined accuracy: {combined_accuracy:.4f}')

# Save all results to a text file
txt_file_path = os.path.join(args['logdir'], f'{model_replace}_accuracy_results_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}_tokenlen{args["token_len"]}_sampleseed{sampling_seed}.txt')
save_accuracy_log(txt_file_path, combined_accuracy, total_count_sum, individual_results)

print(f'Accuracy results saved to {txt_file_path}')
