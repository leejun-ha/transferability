export USE_TORCH=1
task="H3"
models="bert-base-uncased bert-base-chinese bert-base-german-cased 
neuralmind/bert-base-portuguese-cased tohoku-nlp/bert-base-japanese
FacebookAI/roberta-base
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"

# models="FacebookAI/roberta-base"

for model in $models
do
	python len_preprocess.py --task $task \
		--model ${model} \

done