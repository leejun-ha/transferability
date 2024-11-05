export USE_TORCH=1

token_len=256

CUDA_VISIBLE_DEVICES=2 python all_process.py \
    --token_len ${token_len}