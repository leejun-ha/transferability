import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import numpy as np
import copy
import argparse
import tensorboardX
#import torch.utils.tensorboard
#from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time

def create_log_file(args):
    log_dir = os.path.join(args['logdir'], 'log_files')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{args['filename']}_log.txt")
    return log_file

def log_message(log_file, message):
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")


def main():
#device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, default = 'maestro-v1')
    parser.add_argument('--model', type = str, default = 'bert-base-uncased')
    parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'])
    parser.add_argument('--seed', type = int, default = 2020)
    parser.add_argument('--gradient_accumulation', '-a', type = int, default = 2)
    parser.add_argument('--batch_size', '-b', type = int, default = 16)
    parser.add_argument('--epoch', '-e', type = int, default = 20)
    parser.add_argument('--warmup_step', type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--shift', type = int, default = 0)
    # parser.add_argument('--shift_table', type = str, default = './data/maestro-v1.0.0/maestro-v1_bert-base-uncased_token_map.pkl')
    parser.add_argument('--shift_table', type = str, default = '')
    parser.add_argument('--n_gpu', type = int, default ='1')
    parser.add_argument('--ckpt', type = str, default = '')
    
    parser.add_argument('--datadir', type = str, default = './data/pkl')
    parser.add_argument('--logdir', type = str, default = './log')
    parser.add_argument('--savedir', type = str, default = './pth/save_model')
    parser.add_argument('--save_step', type = int, default = 3000)
    parser.add_argument('--filename', type = str)
    parser.add_argument('--postfix', type = str, default = '')
    
    parser.add_argument('--token_len', type = int, default = 128)
    parser.add_argument('--ranking', type = str, default = None)
    parser.add_argument('--oov', type = int, default = 0)
    
    
    args = vars(parser.parse_args())

    # remove "/" in model name
    model_replace = args["model"]
    model_replace = model_replace.replace("/", "_")
    print(model_replace)
    
    token_len = args['token_len']
    ranking = args['ranking']
    if(args['ranking'] == None):
        args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_bert_token_mapping.pkl')
        
    else:
        args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_' + ranking + '_256_token_mapping.pkl')
    
    print(args["shift_table"])
    
    if args['filename'] == None:
        if(args['ranking'] == None):
            if(args['oov'] == 1):
                args['filename'] = f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}_filtered'
            else:
                args['filename'] = f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}'
        else:
            args['filename'] = f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_{ranking}_tokenlen{token_len}'
    if args['shift']!=0:
        args['filename'] += f'_shift{args["shift"]}'
    if args['shift_table']!='':
        args['filename'] += '_table_'
    args['filename'] += args['postfix']
    print(args)
    
    save_dir = os.path.join(args['savedir'], model_replace)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(args['logdir']):
        os.makedirs(args['logdir'])
    train(args = args)

def train(args):
    
    model_replace = args["model"]
    model_replace = model_replace.replace("/", "_")
    
    save_dir = os.path.join(args['savedir'], model_replace)
        
    torch.backends.cudnn.benchmark = True
    
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    ## load a train dataset
    model_name = args['model']
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    batch_size = args['batch_size'] #2 devices
    epoch = args['epoch']
    gradient_accumulation = args['gradient_accumulation']
    token_len = args['token_len']
    
    
    if(args['oov'] == 1):
        data_path = args['datadir']
        data = torch.load(os.path.join(data_path, f'{token_len}_train_data_filtered.pkl'))
    #attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_attention_mask.pkl'))
        label = torch.load(os.path.join(data_path, f'{token_len}_train_label_filtered.pkl'))
        composer2id = torch.load(os.path.join(data_path, 'composer2id_map_filtered.pkl'))
    else:
        data_path = os.path.join(args['datadir'], model_replace )
        data = torch.load(os.path.join(data_path, f'{model_replace}_{token_len}_train_data.pkl'))
    #attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_attention_mask.pkl'))
        label = torch.load(os.path.join(data_path, f'{model_replace}_{token_len}_train_label.pkl'))
        composer2id = torch.load(os.path.join(data_path, 'composer2id_map.pkl'))
    dataset_train = torch.utils.data.TensorDataset(data, label) 
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                                 collate_fn=collate_fn, shuffle=True, pin_memory = True)
    
    num_labels = len(composer2id)


    config = transformers.AutoConfig.from_pretrained(model_name, num_labels = num_labels)
    if args['type'] == 'pretrain':
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                                num_labels = num_labels)#.to(device)
    else:
        model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
    
    if args['ckpt'] != '':
        state_dict = torch.load(args['ckpt'])
        pretrain_config = transformers.AutoConfig.from_pretrained(model_name)
        pretrain_model = transformers.AutoModelForPreTraining.from_pretrained(None,
                                                                           state_dict = state_dict,
                                                                           config = pretrain_config)
        try:
            model.bert = copy.deepcopy(pretrain_model.bert)
        except:
            model.albert = copy.deepcopy(pretrain_model.albert)
        del pretrain_model, state_dict
        print(f"[finetune] Pretrain checkpoint loaded from {args['ckpt']}")

    model.cuda()
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model) 
    optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, args['warmup_step'], len(iterator_train)*epoch/gradient_accumulation)

    
    writer = tensorboardX.SummaryWriter(log_dir=args['logdir'],
                                        filename_suffix=f'_train_{args["task"]}_{args["type"]}_seed{args["seed"]}_shift{args["shift"]}')
    
    log_file = create_log_file(args)
    log_message(log_file, f"Training started with arguments: {args}")
    
    model.train()
    print("Model.train(): ", model.training)
    log_message(log_file, f"Model.train(): {model.training}")
    if args['shift_table']!='':
        shift_table = torch.load(args['shift_table'])
        shift_table.cuda()
        print(f"[finetune] Shift table loaded from {args['shift_table']}")
    logging_step = 50
    global_step = 0
    update_step = 0
    last_step = 0
    #batch_acc = 0
    logging_loss = 0
    tr_loss = 0
    best_loss = float('inf')
    optimizer.zero_grad()
    
    for e in range(epoch):
        for b, (input_ids, labels) in enumerate(tqdm(iterator_train)):
            input_ids = input_ids.cuda(non_blocking=False)
            with torch.no_grad():
                if args['shift_table']!='':
                    input_ids = shift_table(input_ids).to(torch.long).squeeze()
                elif args['shift']!=0:
                    input_ids = torch.remainder(input_ids + args['shift'], model.module.config.vocab_size)
            #attention_mask = attention_mask.cuda(non_blocking=False)
            labels = labels.cuda(non_blocking=False)
            outputs = model(input_ids = input_ids, 
                                 labels = labels)
            if args['n_gpu'] > 1:
                loss = loss.mean()
            loss = outputs.loss
            logits = outputs.logits
            loss = loss/gradient_accumulation
            loss.backward()
            
            tr_loss += loss.item()
            global_step += 1
            if global_step % gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                update_step += 1
                if update_step % logging_step == 0:
                    avg_loss = (tr_loss - logging_loss) / logging_step
                    writer.add_scalar('loss', avg_loss, update_step)
                    log_message(log_file, f"[Step {update_step}] Loss: {avg_loss:.4f}")
                    #print(f"[step {update_step}] loss: {};\batch acc: {batch_acc.item()/batch_size}")
                    logging_loss = tr_loss
                
                    # Save the best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        if args['n_gpu'] > 1:
                            torch.save(model.module.state_dict(), os.path.join(save_dir, args['filename']+'_best.pkl'))
                        else:
                            torch.save(model.state_dict(), os.path.join(save_dir, args['filename']+'_best.pkl'))
                        log_message(log_file, f"New best model saved with loss: {best_loss:.4f}")
                    
                if update_step % args['save_step'] == 0:
                    if args['n_gpu'] > 1:
                        torch.save(model.module.state_dict(), os.path.join(save_dir, args['filename']+'_'+str(update_step)+'.pkl'))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, args['filename']+'_'+str(update_step)+'.pkl'))

    if args['n_gpu'] > 1:
        torch.save(model.module.state_dict(), os.path.join(save_dir, args['filename']+'_'+str(update_step)+'.pkl'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, args['filename']+'_'+str(update_step)+'.pkl'))

if __name__ == '__main__':
    main()
