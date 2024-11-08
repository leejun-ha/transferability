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

from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

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

args = vars(parser.parse_args())

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

# Function to load data for a specific test token length
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

# Evaluate on individual test_token_len datasets
individual_results = {}
individual_accuracies = []
for test_token_len in test_token_lengths:
    print(f"Evaluating on test_token_len {test_token_len}...")
    dataset = load_data(data_path, model_replace, args['split'], test_token_len)
    iterator = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True)
    correct, total = evaluate(model, iterator)
    individual_results[test_token_len] = (correct, total)
    accuracy = correct / total
    individual_accuracies.append(accuracy)
    print(f'Accuracy for test_token_len {test_token_len}: {accuracy:.4f}')

# Calculate combined accuracy
correct_count_sum = sum(correct for correct, _ in individual_results.values())
total_count_sum = sum(total for _, total in individual_results.values())
combined_accuracy = correct_count_sum / total_count_sum

# Calculate average accuracy
average_accuracy = sum(individual_accuracies) / len(individual_accuracies)

print(f'Combined accuracy: {combined_accuracy:.4f}')
print(f'Average accuracy: {average_accuracy:.4f}')

# Update the save_accuracy_log function to include average accuracy
def save_accuracy_log(file_path, combined_acc, average_acc, total_samples, individual_results):
    with open(file_path, 'w') as f:
        f.write(f'Evaluation results:\n\n')
        f.write(f'Combined dataset results:\n')
        f.write(f'Total samples: {total_samples}\n')
        f.write(f'Combined accuracy: {combined_acc:.4f}\n')
        f.write(f'Average accuracy: {average_acc:.4f}\n\n')
        f.write(f'Individual test_token_len results:\n')
        for test_token_len, (correct, total) in individual_results.items():
            acc = correct / total
            f.write(f'test_token_len {test_token_len}:\n')
            f.write(f'  Samples: {total}\n')
            f.write(f'  Correct: {correct}\n')
            f.write(f'  Accuracy: {acc:.4f}\n\n')

# Save all results to a text file
txt_file_path = os.path.join(args['logdir'], f'{model_replace}_accuracy_results_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}_tokenlen{args["token_len"]}.txt')
save_accuracy_log(txt_file_path, combined_accuracy, average_accuracy, total_count_sum, individual_results)

print(f'Accuracy results saved to {txt_file_path}')