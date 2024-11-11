export USE_TORCH=1

models="bert-base-uncased"
# models=FacebookAI/roberta-base

task="localization"
type="scratch"
seed=2020

for model in $models
do
    CUDA_VISIBLE_DEVICES=1 python len_finetune.py --task ${task}\
        --type ${type} \
        --seed ${seed} \
        --model ${model}  \
        --logdir ./log/localization \
        --b 64  \
        --e 15  \
        --n_gpu 1 
done