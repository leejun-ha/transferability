export USE_TORCH=1
task="localization"
split="test"
step="final"
seed=2020
input_lens="64 128 256 384 512"
models="bert-base-uncased bert-base-chinese 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"

for model in $models
do
    for input_len in $input_lens
    do

    CUDA_VISIBLE_DEVICES=3 python len_evaluate.py --task $task \
        --split $split \
        --step $step \
        -b 64 \
        --type pretrain \
        --state_dict ./pth/${task}_bert-base-uncased_pretrain_seed${seed}_len${input_len}_${step}.pkl \
        --logdir ./log/$task/   \
        --current_input_length ${input_len}
    done
done