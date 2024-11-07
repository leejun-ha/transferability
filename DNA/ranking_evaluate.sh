export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=1

task="H4"
models="bert-base-uncased bert-base-chinese 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"
step=2060
seed=2020
split="test"
# model="microsoft/codebert-base-mlm"
# model_state="microsoft_codebert-base-mlm"
ranking="middle"

for model in $models
do
    python evaluate.py --task $task \
        --split $split \
        --step $step \
        -b 64 \
        --type pretrain \
        --seed ${seed} \
        --state_dict ./pth/${task}_ \
        --logdir ./log/$task \
        --datadir ./data \
        --model ${model} \
        --ranking  ${ranking} \
        --shift_table ../Music/shift_table
done
    