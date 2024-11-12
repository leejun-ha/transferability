export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=0

task="H4"
# models="bert-base-chinese 
# bert-base-german-cased neuralmind/bert-base-portuguese-cased 
# tohoku-nlp/bert-base-japanese FacebookAI/roberta-base
# microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# "
models="bert-base-uncased" 
step="final"
seed=2020
split="test"
type="scratch"
# model="microsoft/codebert-base-mlm"
# model_state="microsoft_codebert-base-mlm"
token_lens="64 128 256 384 512"

for model in $models
do
    for token_len in $token_lens
    do
        python len_evaluate.py --task $task \
            --split $split \
            --step ${step} \
            -b 64 \
            --type ${type} \
            --seed ${seed} \
            --state_dict ./pth/${task}_ \
            --logdir ./log/$task \
            --datadir ./data \
            --model ${model} \
            --token_len ${token_len}
    done
done
    