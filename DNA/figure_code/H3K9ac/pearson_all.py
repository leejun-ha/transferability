import os
import json
from scipy.stats import pearsonr

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

# 결과를 저장할 파일 열기
with open('model_results_and_correlations.txt', 'w') as output_file:
    for model_name in models:
        model_folder = f"{model_name}_result"
        
        # token_counts_original.txt 파일 읽기
        token_counts_file = os.path.join(model_folder, "token_counts_original.txt")
        sequence_lengths = [64, 128, 256, 384, 512]
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
        
        # 모델 이름과 성능 결과 저장
        output_file.write(f"{model_name} Results:\n")
        output_file.write("Token Counts:\n")
        for length, count in sequence_counts.items():
            output_file.write(f"{length}: {count}\n")
        output_file.write("\nPerformance Results:\n")
        
        model_performance = []
        for length in sequence_lengths:
            performance = performance_data[str(length)]
            output_file.write(f"{length}: {performance}\n")
            model_performance.append(performance)
        
        # 피어슨 상관계수 계산 및 저장
        correlation, p_value = pearsonr(model_performance, sequence_count_list)
        output_file.write(f"\nPearson Correlation: {correlation:.4f}\n")
        output_file.write(f"P-value: {p_value:.4f}\n")
        output_file.write("\n" + "="*50 + "\n\n")

print("Results and correlations have been saved to 'model_results_and_correlations.txt'")