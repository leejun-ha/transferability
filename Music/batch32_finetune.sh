export USE_TORCH=1

models="wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
microsoft/codebert-base neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
bertin-project/bertin-roberta-base-spanish"

epoch=20
batch=32
#bert-base-uncased bert-base-chinese

for model in $models
do
    CUDA_VISIBLE_DEVICES=2 python finetune.py \
        --model ${model}  \
        --type pretrain \
        -e ${epoch} \
        -b ${batch}

done