export USE_TORCH=1
task="H3"
# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased neuralmind/bert-base-portuguese-cased 
# aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base"

models="FacebookAI/roberta-base"

for model in $models
do
	python preprocess.py --task $task \
		--model ${model} 
done