export USE_TORCH=1
task="maestro-v1"
step=$2
seed=2008
split=$1
python evaluate.py --task $task \
    --split $split \
    --step $step \
    -b 1 \
    --type pretrain \
    --state_dict ./save_model/$task/${task}_bert-base-uncased_pretrain_seed${seed}_table__${step}.pkl \
    --logdir ./log/$task \
    --datadir ./data \
    --shift_table ./data/maestro-v1/${task}_bert-base-uncased_token_map.pkl
