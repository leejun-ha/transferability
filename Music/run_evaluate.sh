export USE_TORCH=1
task="maestro-v1"
models="neulab/codebert-javascript"
# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
# microsoft/codebert-base neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# bertin-project/bertin-roberta-base-spanish" 

step=3000
seed=2020
batch=16
token_len=384

for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python evaluate.py --task $task \
        --split test \
        --step ${step} \
        -b ${batch} \
        --type pretrain \
        --model $model \
        --seed ${seed} \
        --logdir ./log/$task \
        --state_dict ./pth/save_model \
        --datadir ./data/pkl \
        --shift_table ./shift_table \
        --token_len ${token_len} 
        # --shift_table ./data/maestro-v1.0.0/maestro-v1_bert-base-uncased_token_map.pkl\
        
done