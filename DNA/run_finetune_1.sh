export USE_TORCH=1
export CUDA_VISIBLE_DEVICES=0
tasks="H3"
epoch="20"
seed="2020"
# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased neuralmind/bert-base-portuguese-cased 
# aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base"
models="bert-base-multilingual-cased"

for model in $models
do
	for task in $tasks
	do
		python finetune.py --task ${task} \
			--type pretrain \
			--seed $seed \
			--logdir ./log/$task/ \
			--datadir ./data \
			-b 32 \
			-e $epoch \
			--save_step 1000 \
			--n_gpu 1 \
			--model ${model}
	done
done