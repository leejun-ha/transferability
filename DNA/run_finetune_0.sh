export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=2
tasks="H4"
epoch="20"
seed="2020"
models="bert-base-uncased bert-base-chinese 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"
# models="microsoft/codebert-base"
ranking="low"

for model in $models
do
	for task in $tasks
	do
		python finetune.py --task ${task} \
			--type pretrain \
			--seed $seed \
			--logdir ./log/$task/ \
			--datadir ./data \
			-b 64 \
			-e $epoch \
			--save_step 5000 \
			--n_gpu 1 \
			--model ${model}\
			--shift_table ../Music/shift_table\
			--ranking ${ranking}
	done
done