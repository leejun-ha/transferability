export USE_TORCH=1
task="maestro-v1"
epoch=1
seed=2020
python finetune.py --task $task \
	--type pretrain \
    --seed $seed \
    --logdir ./log/$task \
    --datadir ./data \
    -b 32 \
    -a 1 \
    -e $epoch \
    --save_step 1000 \
    --n_gpu 1 \
    --shift_table ./data/maestro-v1/maestro-v1_bert-base-uncased_token_map.pkl