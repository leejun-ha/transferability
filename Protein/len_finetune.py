import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import copy
import argparse
import tensorboardX
import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time

import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, help = "Protrein classification task name")
    parser.add_argument('--model', type = str, default = 'bert-base-uncased', help = "pre-trained model to load")
    parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'], help = "load pre-trained model or not")
    parser.add_argument('--seed', type = int, default = 2020, help = "random seed")
    parser.add_argument('--gradient_accumulation', '-a', type = int, default = 2)
    parser.add_argument('--batch_size', '-b', type = int, default = 16)
    parser.add_argument('--epoch', '-e', type = int, default = 20)
    parser.add_argument('--warmup_step', type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate')
    parser.add_argument('--shift', type = int, default = 0, help = 'the constant c for the "shift c" setting')
    parser.add_argument('--shift_table', type = str, default = '', help = 'the table file for the "random shift" setting')
    parser.add_argument('--rand_embed', action = 'store_true', help = 'run the experiment for randomized embedding')
    parser.add_argument('--n_gpu', type = int)
    parser.add_argument('--ckpt', type = str, default = '', help = 'ckpt file for the experiment of different pre-training steps')

    parser.add_argument('--logdir', type = str, default = './log')
    parser.add_argument('--savedir', type = str, default = './pth')
    parser.add_argument('--save_step', type = int, default = 3000)
    parser.add_argument('--filename', type = str)
    parser.add_argument('--postfix', type = str, default = '')
    
    parser.add_argument('--ranking', type = str, default = '')
    parser.add_argument('--input_lengths', nargs='+', type=int, default=[64, 128, 256, 384, 512], help="Input token lengths to use for fine-tuning")
    parser.add_argument('--log_file', type=str, default='training_log.txt', help="File to save training logs")
    args = vars(parser.parse_args())
    
    logging.basicConfig(filename=args['log_file'], level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Log the arguments
    logging.info(f"Training arguments: {args}")
    model_name = args["model"]
    model_replace = model_name.replace("/", "_")
    if args['filename'] == None:
        args['filename'] = f'{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}'
    if args['shift']!=0:
        args['filename'] += f'_shift{args["shift"]}'
    if args['shift_table']!='':
        args['filename'] += '_table_'
    args['filename'] += args['postfix']
    print(args)
    
    # args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_' + args["ranking"] + '_256_token_mapping.pkl')
    args['filename'] += args["ranking"]
    
    args["data_config"] = f'./config/data/{args["task"]}.json'
    args["sanity_check"] = False

    for input_length in args['input_lengths']:
        print(f"Fine-tuning for input length: {input_length}")
        args['current_input_length'] = input_length
        train(args=args)

def train(args):
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
    data_cfg  = config.DataConfig(args["data_config"])
    cfgs.append(data_cfg)

    model_name = args['model']
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    batch_size = args['batch_size']
    epoch = args['epoch']
    gradient_accumulation = args['gradient_accumulation']
    current_input_length = args['current_input_length']

    # Load the preprocessed dataset for the current input length
    dataset_train = load(data_cfg, "train", alphabet, False)
    dataset_train = mydataset.Seq_dataset(*dataset_train, encoder=alphabet, tokenizer=tokenizer, 
                                          args=args, max_len=current_input_length, 
                                          cache_dir=f'./preprocess_input/{args["task"]}',
                                          split='train')
    
    iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    config = transformers.AutoConfig.from_pretrained(model_name, num_labels=data_cfg.num_labels)
    if args['type'] == 'pretrain':
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=data_cfg.num_labels)
    else:
        model = transformers.AutoModelForSequenceClassification.from_config(config)
    
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
    if args['rand_embed']:
        scratch_model = transformers.AutoModelForSequenceClassification.from_config(config)
        model.bert.embeddings.word_embeddings = copy.deepcopy(scratch_model.bert.embeddings.word_embeddings)
        print(f"[finetune] Word embedding randomized.")

    model.cuda()
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, args['warmup_step'], len(iterator_train)*epoch/gradient_accumulation)

    writer = tensorboardX.SummaryWriter(log_dir=args['logdir'],
                                        filename_suffix=f'_train_{args["task"]}_{args["type"]}_seed{args["seed"]}_shift{args["shift"]}_len{current_input_length}')
    model.train()
    
    print("Model.train(): ", model.training)
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
    optimizer.zero_grad()

    logging.info(f"Model architecture:\n{model}")
    
    for e in range(epoch):
        logging.info(f"Starting epoch {e+1}/{epoch}")
        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator_train)):
            input_ids = input_ids.cuda(non_blocking=False)
            token_type_ids = token_type_ids.cuda(non_blocking=False)
            attention_mask = attention_mask.cuda(non_blocking=False)
            labels = labels.cuda(non_blocking=False)
            
            # ... (keep existing code for shifting input_ids if necessary)

            outputs = model(input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask,
                            labels=labels)
            
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
                    
                    logging.info(f"Step {update_step}: Average loss = {avg_loss:.4f}")
                    
                    logging_loss = tr_loss

                if update_step % args['save_step'] == 0:
                    save_path = os.path.join(args['savedir'], f'{args["filename"]}_len{current_input_length}_{update_step}.pkl')
                    if args['n_gpu'] > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    logging.info(f"Model saved at step {update_step}: {save_path}")

    # Save final model
    save_path = os.path.join(args['savedir'], f'{args["filename"]}_len{current_input_length}_final.pkl')
    if args['n_gpu'] > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    logging.info(f"Final model saved: {save_path}")

if __name__ == '__main__':
    main()