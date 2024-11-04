export USE_TORCH=1

models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"

# model="bert-base-uncased"

epoch=30
batch=16
# token_lens="64 128 256 384 512"
token_len="256"
# rankings="top middle low"
ranking="middle"
oov=0

for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python finetune.py \
        --model ${model}  \
        --type pretrain \
        -e ${epoch} \
        -b ${batch} \
        --token_len ${token_len} \
        --shift_table ./shift_table \
        --ranking ${ranking} \
        --oov ${oov}
done
# for ranking in $rankings
# do
#     CUDA_VISIBLE_DEVICES=0 python finetune.py \
#         --model ${model}  \
#         --type pretrain \
#         -e ${epoch} \
#         -b ${batch} \
#         --token_len ${token_len} \
#         --shift_table ./shift_table \
#         --ranking ${ranking}  \
#         --oov ${oov}
# done