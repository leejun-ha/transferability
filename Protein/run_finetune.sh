export USE_TORCH=1

models="bert-base-uncased bert-base-chinese 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"
task="localization"
type="pretrain"
seed=2020
ranking="top"
for model in $models
do
    CUDA_VISIBLE_DEVICES=3 python finetune.py --task ${task}\
        --type ${type} \
        --seed ${seed} \
        --model ${model}  \
        --logdir ./log/localization \
        --b 64  \
        --e 30   \
        --n_gpu 1 \
        --ranking ${ranking}  \
        --shift_table ../Music/shift_table
done