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
from sklearn.metrics import f1_score, precision_recall_fscore_support

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

parser.add_argument('--logdir', type = str, default = './log')
parser.add_argument('--datadir', type = str)

parser.add_argument('--token_len', type = int, default = 128)
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
test_token_lens = [64, 128, 256, 384, 512]
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
        args["test_token_len"] = args["token_len"]
    else:
        args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_bert_token_mapping.pkl')
else:
    args["shift_table"] = os.path.join(args["shift_table"], model_replace + '_' + ranking + '_256_token_mapping.pkl')

token_len = args["token_len"]    



def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return precision, recall, f1


    # ... (이전 코드는 그대로 유지)
def main():
    
    results = []

    for test_token_len in test_token_lens:
        # ... (데이터 로딩 부분은 그대로 유지)
        data_path = os.path.join(args['datadir'], model_replace)
    data = torch.load(os.path.join(data_path, f'{model_replace}_{test_token_len}_{args["split"]}_data.pkl'))
    #attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_attention_mask.pkl'))
    label = torch.load(os.path.join(data_path, f'{model_replace}_{test_token_len}_{args["split"]}_label.pkl'))

    if(args['oov'] == 1):
        no_oov_data = torch.load(os.path.join(args['datadir'], f'{test_token_len}_{args["split"]}_data_filtered.pkl'))
        no_oov_label = torch.load(os.path.join(args['datadir'], f'{test_token_len}_{args["split"]}_label_filtered.pkl'))
        
        dataset_no_oov = torch.utils.data.TensorDataset(data, label) 

        collate_fn_no_oov = None #dataset.collate_sequences if flag_rnn else None
        iterator_no_oov = torch.utils.data.DataLoader(dataset_no_oov, batch_size=batch_size, 
                                                collate_fn=collate_fn_no_oov, shuffle=False, pin_memory = True)
        
    dataset_dev = torch.utils.data.TensorDataset(data, label) 
    # print(f"Num of Data: {len(dataset_dev)}")
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, 
                                            collate_fn=collate_fn, shuffle=False, pin_memory = True)

    # iterator_dev = zip(data, label) 
    #since here we use a list of songs, 
    #each song contain several segments of 128, use the first dim as batch to vote

    composer2id = torch.load(os.path.join(data_path, 'composer2id_map.pkl'))    
    num_labels = len(composer2id)

    config = transformers.AutoConfig.from_pretrained(model_name, num_labels = num_labels)
    model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
    if args['shift_table'] != '':
        if args['ranking'] == None :
            if( args['oov'] == 1):
                state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}_filtered_table__{args["step"]}.pkl')
            else:    
                state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}_table__{args["step"]}.pkl')
        else:      
            state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_{args["ranking"]}_tokenlen{token_len}_table__{args["step"]}.pkl')
    else:
        state_dict_path = os.path.join(args["state_dict"], model_replace, f'{args["batch_size"]}_{args["task"]}_{model_replace}_{args["type"]}_seed{args["seed"]}_tokenlen{token_len}_{args["step"]}.pkl')
        model.cuda()
        model = model.eval()
        
        if args['shift_table'] != '':
            shift_table = torch.load(args['shift_table']).cuda()
        
        with torch.no_grad():
            dev_loss = 0
            dev_acc = 0
            all_preds = []
            all_labels = []

            for b, (input_ids, labels) in enumerate(tqdm(iterator_dev, total=len(iterator_dev))):
                input_ids = input_ids.to(device)
                if args['shift_table'] != '':
                    input_ids = shift_table(input_ids).long().squeeze()
                labels = labels.to(device)
                logits = model(input_ids=input_ids)
                
                # Handle both scalar and multi-dimensional tensors
                ans = torch.argmax(logits[0], dim=-1)
                if ans.dim() == 0:  # scalar tensor
                    ans = ans.unsqueeze(0)  # convert to 1D tensor
                else:
                    ans = torch.mode(ans).values

                dev_acc = dev_acc + torch.sum(torch.eq(ans, labels)).item()
                
                # Convert to numpy and extend the lists
                all_preds.extend(ans.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Calculate accuracy
        accuracy = dev_acc / len(all_labels)

        # Calculate precision, recall, and F1 score
        precision, recall, f1 = calculate_metrics(all_labels, all_preds)

        print(f"train_token_len:{token_len}")
        print(f"test_token_len:{test_token_len}")
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        results.append({
            'train_token_len': token_len,
            'test_token_len': test_token_len,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        if args['oov'] == 1:
        
        
            with torch.no_grad():
                no_oov_loss = 0
                no_oov_acc = 0
                for b, (input_ids, labels) in enumerate(tqdm(iterator_no_oov, total = len(iterator_no_oov))):
                    input_ids = input_ids.to(device)
                    if args['shift_table']!= '':
                        input_ids = shift_table(input_ids).long().squeeze()
                    #attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                    logits = model(input_ids = input_ids)
                    #loss = loss.mean()*input_ids.shape[0]
                    #dev_loss += loss.item()
                    ans = torch.mode(torch.argmax(logits[0], dim = -1))
                    ans = ans.values
                    no_oov_acc = no_oov_acc + torch.sum(torch.eq(ans, labels)).item()
                #print(f'loss: {dev_loss/len(dataset_dev)}; acc:{dev_acc/len(dataset_dev)}')
                print(f'no_oov_acc:{no_oov_acc/label.shape[0]}')
                #writer.add_scalar(f'{args["split"]}_loss', dev_loss/len(dataset_dev), args['step'])
                # writer.add_scalar(f'{args["split"]}_acc', dev_acc/label.shape[0], args['step'])
                # writer.close()    

    # 결과를 txt 파일로 저장
    output_file = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}_results.txt'
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"Train Token Length: {result['train_token_len']}\n")
            f.write(f"Test Token Length: {result['test_token_len']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write("\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()