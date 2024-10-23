export USE_TORCH=1

# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
# microsoft/codebert-base microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# bertin-project/bertin-roberta-base-spanish"
models="microsoft/codebert-base-mlm"
# models="bert-base-uncased bert-base-chinese"

for model in $models
do
    CUDA_VISIBLE_DEVICES=2 python mapping.py \
        --model ${model}  \

done