import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('./bert-base-uncased_result/token_frequency_ranking.json', 'r') as file:
    data = json.load(file)

# Extract frequencies and sort them in descending order
frequencies = sorted(data.values(), reverse=True)

# Calculate the total number of tokens
total_tokens = len(frequencies)

# Calculate the indices for top 10%, middle 80%, and bottom 10%
top_10_index = int(total_tokens * 0.1)
bottom_10_index = int(total_tokens * 0.9)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the data
plt.plot(range(total_tokens), frequencies, color='blue', alpha=0.7)

# Fill areas for each group
plt.fill_between(range(top_10_index), frequencies[:top_10_index], color='red', alpha=0.3, label='Top 10%')
plt.fill_between(range(top_10_index, bottom_10_index), frequencies[top_10_index:bottom_10_index], color='green', alpha=0.3, label='Middle 80%')
plt.fill_between(range(bottom_10_index, total_tokens), frequencies[bottom_10_index:], color='yellow', alpha=0.3, label='Bottom 10%')

# Set labels and title
plt.xlabel('Token Rank')
plt.ylabel('Frequency')
plt.title('Token Frequency Distribution')

# Set log scale for better visualization
plt.yscale('log')

# Add legend
plt.legend()

# Add text annotations for percentages
plt.text(total_tokens * 0.05, max(frequencies), 'Top 10%', verticalalignment='top')
plt.text(total_tokens * 0.5, np.median(frequencies), 'Middle 80%', verticalalignment='center')
plt.text(total_tokens * 0.95, min(frequencies), 'Bottom 10%', verticalalignment='bottom', horizontalalignment='right')

# Save the plot as a PNG file
plt.savefig('token_frequency_distribution.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()