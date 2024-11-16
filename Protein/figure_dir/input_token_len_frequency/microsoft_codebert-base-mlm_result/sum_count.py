def process_files():
    file_names = [
        '20231101.en_token_counts_original.txt',
        'all_token_counts_original.txt',
        'cc_news_token_counts_original.txt'
    ]
    total_counts = [0] * 513  # Initialize a list to store sums for indices 0-512

    for file_name in file_names:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    index = int(parts[0])
                    value = int(parts[1])
                    if 0 <= index <= 512:
                        total_counts[index] += value

    # Write the results to token_counts_original.txt
    with open('token_counts_original.txt', 'w') as f:
        for i, count in enumerate(total_counts):
            if i <= 512:
                f.write(f"{i}: {count}\n")

    print("Processing complete. Results written to token_counts_original.txt")

# Run the function
process_files()