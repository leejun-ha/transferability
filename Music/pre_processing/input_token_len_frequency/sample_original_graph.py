import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read data from the text file (paste-2.txt)
data = {}
with open('/home/junha/transferability/Music/pre_processing/input_token_len_frequency/neulab_codebert-c_result/token_counts_original.txt.txt', 'r') as file:
    for line in file:
        index, value = line.split(':')
        data[int(index)] = int(value)

# Step 2: Prepare data for plotting
token_lengths = list(data.keys())
counts = list(data.values())

# Step 3: Create the plot
plt.figure(figsize=(10, 6))
plt.bar(token_lengths, counts)

# Step 4: Set log scale for y-axis
plt.yscale('log')

# Step 5: Add labels and title
plt.xlabel('Input Token Length')
plt.ylabel('Count (log scale)')
plt.title('Distribution of Input Token Lengths - neulab_codebert-c')

# Display min and max counts on the plot
min_count = min(counts)
max_count = max(counts)
plt.text(0.1, 0.9, f'Min count: {min_count}', transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f'Max count: {max_count}', transform=plt.gca().transAxes)

# Show plot
plt.show()