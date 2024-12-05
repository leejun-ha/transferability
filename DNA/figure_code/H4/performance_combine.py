import matplotlib.pyplot as plt
import json

sequence_lengths = [64, 128, 256, 384, 512]

plt.figure(figsize=(12, 8))

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

for model_name in models:
    performance_file = f"{model_name}_result/performance_results.json"
    with open(performance_file, 'r') as f:
        performance_data = json.load(f)
    
    performances = [performance_data[str(length)] for length in sequence_lengths]
    plt.plot(sequence_lengths, performances, label=model_name, alpha=0.5)

plt.xlabel('Sequence Length')
plt.ylabel('Performance')
plt.title('Performance vs Sequence Length for All Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()