export USE_TORCH=1

models="bert-base-uncased bert-base-chinese bert-base-german-cased 
neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c"


# model=""

epoch=30
batch=16
# token_lens="64 512"
token_len="256"
# rankings="top middle low"

oov=1
oov_ver=5
oov_method=skip

for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python oov_finetune.py \
        --model ${model}  \
        --type pretrain \
        -e ${epoch} \
        -b ${batch} \
        --token_len ${token_len} \
        --shift_table ./shift_table \
        --oov ${oov} \
        --oov_version ${oov_ver} \
        --oov_method ${oov_method}
done
