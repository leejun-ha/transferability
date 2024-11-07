export USE_TORCH=1
task="localization"
models="bert-base-uncased bert-base-chinese 
bert-base-german-cased neuralmind/bert-base-portuguese-cased 
tohoku-nlp/bert-base-japanese
microsoft/codebert-base-mlm neulab/codebert-javascript neulab/codebert-java neulab/codebert-python neulab/codebert-c
"
# model="FacebookAI/roberta-base"

splits="train dev test"

for model in $models
do
    for split in $splits
	do
        python save_feature.py --task $task \
            --model ${model} \
            --split ${split}
    done
done