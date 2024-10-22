export USE_TORCH=1

# models="wietsedv/bert-base-dutch-cased	neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert tohoku-nlp/bert-base-japanese kykim/bert-kor-base 
# microsoft/codebert-base neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# bertin-project/bertin-roberta-base-spanish"

models="neulab/codebert-javascript"

# "bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased" 
token_len=128

for model in $models
do
	CUDA_VISIBLE_DEVICES=2 python preprocess.py \
		--model ${model} \
		--token_len ${token_len}
done