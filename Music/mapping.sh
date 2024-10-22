export USE_TORCH=1

# models="wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
# microsoft/codebert-base neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# bertin-project/bertin-roberta-base-spanish"
models="bert-base-multilingual-uncased bert-base-multilingual-cased 
bert-base-german-cased"
# models="bert-base-uncased bert-base-chinese"

for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python mapping.py \
        --model ${model}  \

done