export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=1
task=$1
step=4000
seed=2020
split="test"
model="microsoft/codebert-base-mlm"
model_state="microsoft_codebert-base-mlm"
python evaluate.py --task $task \
    --split $split \
    --step $step \
    -b 64 \
    --type pretrain \
    --state_dict ./save_model/${task}_${model_state}_pretrain_seed${seed}_${step}.pkl \
    --logdir ./log/$task \
    --datadir ./data \
    --model $model 