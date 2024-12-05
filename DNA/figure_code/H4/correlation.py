import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 모델 이름 리스트
models = [
    'bert-base-uncased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'neuralmind_bert-base-portuguese-cased',
    'tohoku-nlp_bert-base-japanese',
    'microsoft_codebert-base-mlm',
    'neulab_codebert-javascript',
    'neulab_codebert-java',
    'neulab_codebert-python',
    'neulab_codebert-c'
]

# 관심 있는 시퀀스 길이 범위
sequence_lengths = [256, 384, 512]

# 결과를 저장할 딕셔너리
results = {}

for model_name in models:
    model_folder = f"{model_name}_result"
    
    # token_counts_original.txt 파일 읽기
    token_counts_file = os.path.join(model_folder, "token_counts_original.txt")
    sequence_counts = {}
    
    with open(token_counts_file, 'r') as f:
        for line in f:
            length, count = map(int, line.strip().split(': '))
            if length in sequence_lengths:
                sequence_counts[length] = count
    
    # 시퀀스 길이에 맞는 데이터 수 리스트 생성
    sequence_count_list = [sequence_counts[length] for length in sequence_lengths]
    
    # 성능 결과 파일 읽기
    performance_file = os.path.join(model_folder, "performance_results.json")
    with open(performance_file, 'r') as f:
        performance_data = json.load(f)
    
    # 시퀀스 길이에 맞는 성능 리스트 생성
    model_performance = [performance_data[str(length)] for length in sequence_lengths]
    
    # 피어슨 상관계수 계산
    pearson_correlation, pearson_p_value = pearsonr(model_performance, sequence_count_list)
    
    # 스피어만 상관계수 계산
    spearman_correlation, spearman_p_value = spearmanr(model_performance, sequence_count_list)
    
    # 결과 저장
    results[model_name] = {
        'pearson_correlation': pearson_correlation,
        'pearson_p_value': pearson_p_value,
        'spearman_correlation': spearman_correlation,
        'spearman_p_value': spearman_p_value
    }

# 결과를 파일에 저장
with open('model_correlations.txt', 'w') as output_file:
    for model_name, result in results.items():
        output_file.write(f"{model_name}:\n")
        output_file.write(f"  Pearson Correlation: {result['pearson_correlation']:.4f}\n")
        output_file.write(f"  Pearson P-value: {result['pearson_p_value']:.4f}\n")
        output_file.write(f"  Spearman Correlation: {result['spearman_correlation']:.4f}\n")
        output_file.write(f"  Spearman P-value: {result['spearman_p_value']:.4f}\n")
        output_file.write("\n")

print("Results have been saved to 'model_correlations.txt'")