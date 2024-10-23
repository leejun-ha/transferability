export USE_TORCH=1

# models="bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased wietsedv/bert-base-dutch-cased  neuralmind/bert-base-portuguese-cased	aubmindlab/bert-base-arabert 
# tohoku-nlp/bert-base-japanese kykim/bert-kor-base bertin-project/bertin-roberta-base-spanish 
# microsoft/codebert-base microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
# "

model="microsoft/codebert-base"

# "bert-base-uncased bert-base-chinese bert-base-multilingual-uncased bert-base-multilingual-cased 
# bert-base-german-cased" 
token_lens="64 128 256 384 512"

for token_len in $token_lens
do
	CUDA_VISIBLE_DEVICES=2 python preprocess.py \
		--model ${model} \
		--token_len ${token_len}
done