import os
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--token_len', type=int, default=256)
parser.add_argument('--data_dir', type=str, default='../data/pkl')
args = parser.parse_args()

# Initialize statistics storage
statistics = []

# Define methods
methods = ['skip', 'mask']

# Iterate through each version
for version in range(1, 45):
    for method in methods:
        version_path = os.path.join(args.data_dir, f'version_{version}_{method}')
        # Try to load train data
        try:
            train_data = torch.load(os.path.join(version_path, f'{args.token_len}_train_data_filtered.pkl'))

            # For 'skip' method, OOV tokens are not present in the data
            if method == 'skip':
                oov_count = 0
                oov_data_count = 0
            else:  # For 'mask' method
                # Assuming OOV token is represented by the smallest value in the tensor
                oov_token = train_data.min().item()
                # Count number of OOV tokens
                oov_count = (train_data == oov_token).sum().item()
                # Count number of sequences containing at least one OOV token
                oov_data_count = ((train_data == oov_token).sum(dim=1) > 0).sum().item()

            # Store statistics
            statistics.append({
                'version': version, 
                'method': method, 
                'total_sequences': len(train_data),
                'oov_token_count': oov_count,
                'sequences_with_oov': oov_data_count,
                'oov_token_percentage': oov_count / (len(train_data) * args.token_len) * 100,
                'sequences_with_oov_percentage': oov_data_count / len(train_data) * 100
            })
            
        except FileNotFoundError:
            print(f"Data for version {version} method {method} not found.")

# Create DataFrame from statistics
statistics_df = pd.DataFrame(statistics)

# Display the results
print(statistics_df)

# Optionally, save the statistics to a CSV file
statistics_df.to_csv(f'oov_statistics_{args.token_len}.csv', index=False)

print(f"Statistics saved to oov_statistics_{args.token_len}.csv")