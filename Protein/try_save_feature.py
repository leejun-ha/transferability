import src.config as config
from src.data.alphabets import Protein
import src.data.localization as localization
import src.data.fluorescence as fluorescence
import src.data.solubility as solubility
import src.data.secstr as secstr
import src.data.stability as ss
import src.data.transmembrane as transmembrane

import src.data.mydataset as mydataset
from src.utils import set_seeds
import os
import torch
import argparse
import transformers
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--savedir', type=str, default='./preprocess_input')
parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'])
args = vars(parser.parse_args())

print(args)

LOAD_FUNCTION_MAP = {
    "localization": localization.load_localization,
    "transmembrane": transmembrane.load_transmembrane,
    "secstr": secstr.load_secstr,
    "solubility": solubility.load_solubility,
    "stability": ss.load_stability,
    "fluorescence": fluorescence.load_fluorescence
}

load = LOAD_FUNCTION_MAP[args['task']]

args['savedir'] = os.path.join(args['savedir'], args['task'])
os.makedirs(args['savedir'], exist_ok=True)

set_seeds(args['seed'])
args['data_config'] = f'./config/data/{args["task"]}.json'
args['sanity_check'] = False
alphabet = Protein()
data_cfg = config.DataConfig(args['data_config'])

model_name = args['model']
model_replace = model_name.replace("/", "_")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Define the input token lengths
input_token_lengths = [64, 128, 256, 384, 512]

for max_len in input_token_lengths:
    print(f"Processing for max_len: {max_len}")
    
    dataset = load(data_cfg, args['split'], alphabet, False)
    dataset = mydataset.Loc_dataset(*dataset, encoder=alphabet, tokenizer=tokenizer, 
                                    args=args, max_len=max_len, cache_dir=f'./preprocess_input/{args["task"]}',
                                    split=args['split'])
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator)):
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)

    input_ids_list = torch.cat(input_ids_list, dim=0)
    token_type_ids_list = torch.cat(token_type_ids_list, dim=0)
    attention_mask_list = torch.cat(attention_mask_list, dim=0)

    torch.save(input_ids_list, os.path.join(args['savedir'], f'cached_{args["split"]}_input_ids_{args["task"]}_{model_replace}_{max_len}.pkl'))
    torch.save(token_type_ids_list, os.path.join(args['savedir'], f'cached_{args["split"]}_token_type_{args["task"]}_{model_replace}_{max_len}.pkl'))
    torch.save(attention_mask_list, os.path.join(args['savedir'], f'cached_{args["split"]}_att_mask_{args["task"]}_{model_replace}_{max_len}.pkl'))

print("All datasets processed and saved.")