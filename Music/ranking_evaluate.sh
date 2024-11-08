export USE_TORCH=1
task="maestro-v1"

models="bert-base-uncased bert-base-chinese
bert-base-german-cased neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c"
# model="bert-base-uncased"
# models="neulab/codebert-python"
step="best"
seed=2020
batch=16
token_len="256"
# token_lens="256"
ranking="align"
oov=0
for model in $models
do
    CUDA_VISIBLE_DEVICES=1 python evaluate.py --task $task \
        --split test \
        --step ${step} \
        -b ${batch} \
        --type pretrain \
        --model ${model} \
        --seed ${seed} \
        --logdir ./log/$task \
        --state_dict ./pth/save_model \
        --datadir ./data/pkl \
        --shift_table ./shift_table/freq_align \
        --token_len ${token_len} \
        --ranking ${ranking}  \
        --oov ${oov}
        # --shift_table ./data/maestro-v1.0.0/maestro-v1_bert-base-uncased_token_map.pkl\
        
done
