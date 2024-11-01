export USE_TORCH=1

# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
# microsoft/codebert-base neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# bertin-project/bertin-roberta-base-spanish"

models="bert-base-uncased"

epoch=15
batch=16
token_len=384

for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python finetune.py \
        --model ${model}  \
        --type pretrain \
        -e ${epoch} \
        -b ${batch} \
        --token_len ${token_len} \
        --shift_table ./shift_table

done