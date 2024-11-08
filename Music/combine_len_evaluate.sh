export USE_TORCH=1

task="maestro-v1"

# models="bert-base-uncased bert-base-chinese
# bert-base-german-cased neuralmind/bert-base-portuguese-cased 
# tohoku-nlp/bert-base-japanese
# microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# "

models="microsoft/codebert-base-mlm neulab/codebert-java tohoku-nlp/bert-base-japanese"

# model="bert-base-multilingual-cased"
step="best"
seed=2020
batch=16
token_lens="64 128 256 384 512"

oov="0"
for model in $models
do
    for token_len in $token_lens
    do

        CUDA_VISIBLE_DEVICES=0 python combine_len_evaluate.py \
            --task ${task} \
            --split test \
            --step ${step} \
            -b ${batch} \
            --type pretrain \
            --model ${model} \
            --seed ${seed} \
            --state_dict ./pth/save_model \
            --datadir ./data/pkl \
            --shift_table ./shift_table \
            --token_len ${token_len} \
            --oov ${oov}

    done
done