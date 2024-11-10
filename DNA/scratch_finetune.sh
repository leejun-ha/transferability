export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=0
tasks="H4"
epoch="20"
seed="2020"
models="bert-base-uncased"
# models= FacebookAI/roberta-base

for model in $models
do
	for task in $tasks
	do
		python len_finetune.py --task ${task} \
			--type scratch \
			--seed $seed \
			--logdir ./log/$task/ \
			--datadir ./data \
			-b 64 \
			-e $epoch \
			--save_step 5000 \
			--n_gpu 1 \
			--model ${model}
	done
done