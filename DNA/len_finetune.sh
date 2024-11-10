export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=3
tasks="H4"
epoch="20"
seed="2020"
models="bert-base-uncased microsoft/codebert-base-mlm bert-base-chinese 
neulab/codebert-python bert-base-german-cased neulab/codebert-c

  
"
# models= neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
# neulab/codebert-javascript  neulab/codebert-java

for model in $models
do
	for task in $tasks
	do
		python len_finetune.py --task ${task} \
			--type pretrain \
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