import src.config as config
from src.data.alphabets import Protein
import src.data.localization as localization
import src.data.fluorescence as fluorescence
import src.data.solubility as solubility
import src.data.secstr as secstr
import src.data.stability as ss
import src.data.transmembrane as transmembrane
import src.data.mydataset as mydataset
from src.utils import Print, set_seeds, set_output, load_models

import os
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import tensorboardX

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--type', type=str, choices=['pretrain', 'scratch'])
parser.add_argument('--split', type=str, choices=['dev', 'test'], default='dev')
parser.add_argument('--state_dict', type=str)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--shift_table', type=str, default='')
parser.add_argument('--step', type=str, help='load the checkpoints of different fine-tuning steps')

parser.add_argument('--logdir', type=str, default='./log')
parser.add_argument('--current_input_length', type=int, default='')
args = vars(parser.parse_args())
print(args)

args["data_config"] = f'./config/data/{args["task"]}.json'
args["sanity_check"] = False

set_seeds(args['seed'])
torch.backends.cudnn.benchmark = True

LOAD_FUNCTION_MAP = {
    "localization": localization.load_localization,
    "transmembrane": transmembrane.load_transmembrane,
    "secstr": secstr.load_secstr,
    "solubility": solubility.load_solubility,
    "stability": ss.load_stability,
    "fluorescence": fluorescence.load_fluorescence
}
load = LOAD_FUNCTION_MAP[args['task']]

alphabet = Protein()
cfgs = []
data_cfg = config.DataConfig(args["data_config"])
cfgs.append(data_cfg)

model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
batch_size = args['batch_size']

config = transformers.AutoConfig.from_pretrained(model_name, num_labels=data_cfg.num_labels)
model = transformers.AutoModelForSequenceClassification.from_config(config)
model.load_state_dict(torch.load(args["state_dict"]))
model.cuda()

if args['shift_table'] != '':
    shift_table = torch.load(args['shift_table']).cuda()

writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], 
                                    filename_suffix=f'_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}')

def save_results(results, filename):
    with open(filename, 'w') as f:
        f.write("Evaluation results:\n\n")
        
        f.write("Combined dataset results:\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Combined accuracy: {results['combined_accuracy']:.4f}\n")
        f.write(f"Average accuracy: {results['average_accuracy']:.4f}\n\n")
        
        f.write("Individual test_token_len results:\n")
        for token_len, data in results['individual_results'].items():
            f.write(f"test_token_len {token_len}:\n")
            f.write(f"  Samples: {data['samples']}\n")
            f.write(f"  Correct: {data['correct']}\n")
            f.write(f"  Accuracy: {data['accuracy']:.4f}\n\n")

def evaluate_model(dataset, iterator, token_len):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator)):
            input_ids = input_ids.to(device)
            if args['shift_table'] != '':
                input_ids = shift_table(input_ids).long().squeeze()
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask,
                            labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return {
        'samples': total,
        'correct': correct,
        'accuracy': accuracy
    }

# Test for different token lengths
token_lengths = [64, 128, 256, 384, 512]  # Add or modify token lengths as needed
results = {
    'total_samples': 0,
    'combined_accuracy': 0,
    'average_accuracy': 0,
    'individual_results': {}
}

for max_len in token_lengths:
    print(f"\nTesting with max token length: {max_len}")
    
    dataset_dev = load(data_cfg, args['split'], alphabet, False)
    dataset_dev = mydataset.Seq_dataset(*dataset_dev, encoder=alphabet, tokenizer=tokenizer, 
                                        args=args, max_len=max_len, cache_dir=f'./preprocess_input/{args["task"]}',
                                        split=args['split'])
    collate_fn = None
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)

    result = evaluate_model(dataset_dev, iterator_dev, max_len)
    results['individual_results'][max_len] = result
    results['total_samples'] += result['samples']
    results['combined_accuracy'] += result['correct']

results['combined_accuracy'] /= results['total_samples']
results['average_accuracy'] = sum(r['accuracy'] for r in results['individual_results'].values()) / len(results['individual_results'])

# Save results to a text file
save_results(results, f'evaluation_results_{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}.txt')

print("Evaluation complete. Results saved to text file.")