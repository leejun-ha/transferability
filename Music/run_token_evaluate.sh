export USE_TORCH=1
task="maestro-v1"

# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
# microsoft/codebert-base microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c"
model="bert-base-uncased"
step="best"
seed=2020
batch=16
# token_lens="64 128 256 384 512"
token_len="256"
rankings="top middle low"

for ranking in $rankings
do
    CUDA_VISIBLE_DEVICES=3 python evaluate.py --task $task \
        --split test \
        --step ${step} \
        -b ${batch} \
        --type pretrain \
        --model ${model} \
        --seed ${seed} \
        --logdir ./log/$task \
        --state_dict ./pth/save_model \
        --datadir ./data/pkl \
        --shift_table ./shift_table \
        --token_len ${token_len} \
        --ranking ${ranking}
        # --shift_table ./data/maestro-v1.0.0/maestro-v1_bert-base-uncased_token_map.pkl\
        
done