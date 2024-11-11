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

import logging

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--model', type = str, default = 'bert-base-multilingual-uncased')
parser.add_argument('--seed', type = int, default = 2020)
parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'])
parser.add_argument('--split', type = str, choices=['dev', 'test'], default='dev')
parser.add_argument('--state_dict', type = str)
parser.add_argument('--batch_size', '-b', type = int, default = 32)
parser.add_argument('--shift_table', type = str, default = '')
parser.add_argument('--step', type = str)

parser.add_argument('--logdir', type = str, default = './log')
parser.add_argument('--datadir', type = str)
parser.add_argument('--ranking', type = str, default = '')
parser.add_argument('--token_len', type = str, default = '')
args = vars(parser.parse_args())

#if args['filename'] == None:
#    args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}'
print(args)
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])


torch.backends.cudnn.benchmark = True

## load a dev dataset
model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
batch_size = args['batch_size'] #2 devices

model_replace = model_name
model_replace = model_replace.replace("/", "_")
print(model_replace)
test_lens = [64, 128, 256, 384, 512]
# args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_' + args["ranking"] + '_256_token_mapping.pkl')
args["state_dict"] += model_replace + "_" + "pretrain_seed" + str(args["seed"]) +"_" + "length" + args["token_len"] + "_" + str(args["step"]) + ".pkl"

individual_results = {}

for testlen in test_lens: 
    data_path = os.path.join(args['datadir'], args['task'])
    data = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_data_{testlen}.pkl'))
    attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_attention_mask_{testlen}.pkl'))
    label = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_label_{testlen}.pkl'))

    #train split
    # data_num = data.shape[0]
    # if args['split'] == 'dev':
    #     split_start = int(data_num*0.9)
    #     split_end = int(data_num*0.95)
    # elif args['split'] == 'test':
    #     split_start = int(data_num*0.95)
    #     split_end = int(data_num)
    # data = data[split_start:split_end]
    # attention_mask = attention_mask[split_start:split_end]
    # label = label[split_start:split_end]

    dataset_dev = torch.utils.data.TensorDataset(data, attention_mask, label) 
    print(f"Num of Data: {len(dataset_dev)}")
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, 
                                            collate_fn=collate_fn, shuffle=True, pin_memory = True)

    if args["task"] == 'splice':
        num_labels = 3
    else:
        num_labels = 2

    config = transformers.AutoConfig.from_pretrained(model_name, num_labels = num_labels)
    model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
    model.load_state_dict(torch.load(args["state_dict"]))
    model.cuda()
    #model = torch.nn.DataParallel(model)
    if args['shift_table'] != '':
        shift_table = torch.load(args['shift_table']).cuda()

    # writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], 
    #                                     filename_suffix=f'_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}')
    model = model.eval()

    def log_evaluation_results(total_samples, average_accuracy, individual_results):
        log_filename = f'evaluation_results_{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}_tokenlen{args["token_len"]}.txt'
        
        logging.basicConfig(filename=log_filename, level=logging.INFO, 
                            format='%(message)s', filemode='w')
        logger = logging.getLogger()
        
        logger.info("Evaluation results:\n")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Average accuracy: {average_accuracy:.4f}\n")
        
        logger.info("Individual test_token_len results:")
        for testlen, results in individual_results.items():
            logger.info(f"test_token_len {testlen}:")
            logger.info(f"  Samples: {results['samples']}")
            logger.info(f"  Correct: {results['correct']}")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}\n")

    with torch.no_grad():
        dev_loss = 0
        dev_acc = 0
        for b, (input_ids, attention_mask, labels) in enumerate(tqdm(iterator_dev)):
            input_ids = input_ids.to(device)
            if args['shift_table']!= '':
                input_ids = shift_table(input_ids).long().squeeze()
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids = input_ids, 
                                attention_mask = attention_mask,
                                labels = labels)
            loss = outputs.loss
            logits = outputs.logits
            loss = loss.mean()*input_ids.shape[0]
            dev_loss += loss.item()
            ans = torch.argmax(logits, dim = -1)
            dev_acc = dev_acc + torch.sum(torch.eq(ans, labels)).item()
        print(f'loss: {dev_loss/len(dataset_dev)}; acc:{dev_acc/len(dataset_dev)}')
        # writer.add_scalar(f'{args["split"]}_loss', dev_loss/len(dataset_dev), args['step'])
        # writer.add_scalar(f'{args["split"]}_acc', dev_acc/len(dataset_dev), args['step'])
        # writer.close()
    
