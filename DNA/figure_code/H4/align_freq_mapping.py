
import json
import pickle
from collections import OrderedDict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_sorted_tokens(freq_dict):
    return sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

def create_percentile_mapping(tokens):
    total = len(tokens)
    return {token: i / total for i, (token, _) in enumerate(tokens)}

def main():
    # Load frequency data
    bert_freq = load_json('./bert-base-uncased_result/token_frequency_ranking.json')
    maestro_freq = load_json('../maestro_v1_train_token_freq_original.json')

    # Sort tokens by frequency
    bert_tokens = get_sorted_tokens(bert_freq)
    maestro_tokens = get_sorted_tokens(maestro_freq)

    # Create percentile mappings
    bert_percentiles = create_percentile_mapping(bert_tokens)
    maestro_percentiles = create_percentile_mapping(maestro_tokens)

    # Create the final mapping dictionary
    final_mapping = OrderedDict()
    for i in range(256):
        if i < 128:
            final_mapping[i] = i  # Keep original mapping for 0-127
        else:
            maestro_index = i - 128
            if maestro_index < len(maestro_tokens):
                maestro_token, _ = maestro_tokens[maestro_index]
                maestro_percentile = maestro_percentiles[maestro_token]
                
                # Find the closest BERT token by percentile
                closest_bert_token = min(bert_percentiles.items(), key=lambda x: abs(x[1] - maestro_percentile))[0]
                bert_index = bert_tokens.index((closest_bert_token, bert_freq[closest_bert_token]))
                
                final_mapping[i] = bert_index
            else:
                final_mapping[i] = i  # If we run out of Maestro tokens, keep the original index

    # Save the mapping
    with open('token_mapping.pkl', 'wb') as f:
        pickle.dump(final_mapping, f)

    # Create log file
    with open('token_mapping_log.txt', 'w') as log_file:
        log_file.write("Token Mapping Log\n")
        log_file.write(f"Total BERT Tokens: {len(bert_freq)}\n")
        log_file.write(f"Total Maestro Tokens: {len(maestro_freq)}\n\n")
        log_file.write("Mapping for indices 128-255:\n\n")
        log_file.write("Index | Maestro Token | Maestro % | Maestro Count | BERT Token | BERT % | BERT Count\n")
        log_file.write("-" * 90 + "\n")

        for i in range(128, 256):
            maestro_index = i - 128
            if maestro_index < len(maestro_tokens):
                maestro_token, maestro_count = maestro_tokens[maestro_index]
                maestro_percent = maestro_percentiles[maestro_token]
                
                bert_index = final_mapping[i]
                bert_token, bert_count = bert_tokens[bert_index]
                bert_percent = bert_percentiles[bert_token]

                log_file.write(f"{i:5d} | {maestro_token:13s} | {maestro_percent:8.2%} | {maestro_count:12d} | {bert_token:10s} | {bert_percent:6.2%} | {bert_count:10d}\n")
            else:
                log_file.write(f"{i:5d} | {'N/A':13s} | {'N/A':8s} | {'N/A':12s} | {'N/A':10s} | {'N/A':6s} | {'N/A':10s}\n")

    print(f"Mapping saved to token_mapping.pkl")
    print(f"Log file saved to token_mapping_log.txt")

if __name__ == "__main__":
    main()

