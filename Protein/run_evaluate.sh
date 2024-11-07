export USE_TORCH=1
task=$1
split=$2
step=6701
seed=2020
python evaluate.py --task $task \
    --split $split \
    --step $step \
    -b 64 \
    --type pretrain \
    --state_dict ./save_model/${task}_bert-base-uncased_pretrain_seed${seed}_table__${step}.pkl \
    --logdir ./log/$task/
