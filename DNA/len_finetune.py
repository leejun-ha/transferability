import os
import torch
import random
import numpy as np
import argparse
import tensorboardX
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--type', type=str, choices=['pretrain', 'scratch'])
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--gradient_accumulation', '-a', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--warmup_step', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_gpu', type=int)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--savedir', type=str, default='./pth')
    parser.add_argument('--save_step', type=int, default=3000)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--postfix', type=str, default='')
    args = parser.parse_args()

    if args.filename is None:
        model_replace = args.model.replace("/", "_")
        args.filename = f'{args.task}_{model_replace}_{args.type}_seed{args.seed}{args.postfix}'

    print(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    train(args=vars(args))

def train(args):
    torch.backends.cudnn.benchmark = True
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    model_name = args['model']
    batch_size = args['batch_size']
    epoch = args['epoch']
    gradient_accumulation = args['gradient_accumulation']

    model_replace = args["model"].replace("/", "_")
    data_path = os.path.join(args['datadir'], args['task'])

    input_lengths = [64, 128, 256, 384, 512]

    for length in input_lengths:
        print(f"Training model for input length: {length}")
        
        # Load data for the current input length
        data = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_data_{length}.pkl'))
        attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_attention_mask_{length}.pkl'))
        label = torch.load(os.path.join(data_path, f'{args["task"]}_{model_replace}_label_{length}.pkl'))
        
        # Train split (90% of data)
        data_num = data.shape[0]
        train_data_num = int(data_num * 0.9)
        data = data[:train_data_num]
        attention_mask = attention_mask[:train_data_num]
        label = label[:train_data_num]
        
        dataset_train = torch.utils.data.TensorDataset(data, attention_mask, label)
        iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

        if args["task"] == 'splice':
            num_labels = 3
        else:
            num_labels = 2

        config = transformers.AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        
        # Modify the config to use our custom vocabulary
        # config.vocab_size = 7  # A, T, C, G, [PAD], [CLS], [SEP]
        
        if args['type'] == 'pretrain':
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            model = transformers.AutoModelForSequenceClassification.from_config(config)

        if args['ckpt'] != '':
            state_dict = torch.load(args['ckpt'])
            model.load_state_dict(state_dict)
            print(f"[finetune] Checkpoint loaded from {args['ckpt']}")

        model.cuda()

        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        scheduler = get_linear_schedule_with_warmup(optimizer, args['warmup_step'], len(iterator_train) * epoch // gradient_accumulation)

        writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], filename_suffix=f'_train_{args["task"]}_{args["type"]}_seed{args["seed"]}_length{length}')

        model.train()
        print("Model.train(): ", model.training)

        logging_step = 50
        global_step = 0
        update_step = 0
        logging_loss = 0
        tr_loss = 0
        optimizer.zero_grad()

        for e in range(epoch):
            for b, (input_ids, attention_mask, labels) in enumerate(tqdm(iterator_train)):
                input_ids = input_ids.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                if args['n_gpu'] > 1:
                    loss = loss.mean()
                
                loss = loss / gradient_accumulation
                loss.backward()
                tr_loss += loss.item()

                global_step += 1
                if global_step % gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_step += 1

                    if update_step % logging_step == 0:
                        writer.add_scalar('loss', (tr_loss - logging_loss) / logging_step, update_step)
                        print(f"step: {update_step} loss: {(tr_loss - logging_loss) / logging_step}")
                        logging_loss = tr_loss

                    if update_step % args['save_step'] == 0:
                        if args['n_gpu'] > 1:
                            torch.save(model.module.state_dict(), os.path.join(args['savedir'], f"{args['filename']}_length{length}_{update_step}.pkl"))
                        else:
                            torch.save(model.state_dict(), os.path.join(args['savedir'], f"{args['filename']}_length{length}_{update_step}.pkl"))

        if args['n_gpu'] > 1:
            torch.save(model.module.state_dict(), os.path.join(args['savedir'], f"{args['filename']}_length{length}_final.pkl"))
        else:
            torch.save(model.state_dict(), os.path.join(args['savedir'], f"{args['filename']}_length{length}_final.pkl"))

        print(f"Training completed for input length: {length}")

if __name__ == '__main__':
    main()