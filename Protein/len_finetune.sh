export USE_TORCH=1

models="bert-base-uncased microsoft/codebert-base-mlm bert-base-chinese 
neulab/codebert-python bert-base-german-cased neulab/codebert-c
"
# models= neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
# neulab/codebert-javascript  neulab/codebert-java

task="localization"
type="pretrain"
seed=2020

for model in $models
do
    CUDA_VISIBLE_DEVICES=1 python len_finetune.py --task ${task}\
        --type ${type} \
        --seed ${seed} \
        --model ${model}  \
        --logdir ./log/localization \
        --b 64  \
        --e 30   \
        --n_gpu 1 
done