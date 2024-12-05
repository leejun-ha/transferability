import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['H4', 'H3K9ac', 'Protein', 'Music']

# Results: only BERT results [pre-trained, scratch]
results = {
    'H4':     [0.8557, 0.8388],
    'H3K9ac': [0.7852, 0.7655],
    'Protein':[0.6775, 0.6438],
    'Music':  [0.5752, 0.5325]
}

# Set up bar positions
width = 0.35  # Bar width
spacing = 0.65  # Space between task groups

# Calculate positions for bars
x = np.arange(len(tasks))

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for pre-trained tasks
pretrained_colors = {
    'H4': '#20B2AA',       # Royal Blue
    'H3K9ac': '#48D1CC',   # Light Sea Green
    'Protein': '#6A5ACD',  # Slate Blue
    'Music': '#4169E1'    # Medium Turquoise
}

# Plot bars
pretrained_bars = {}  # To store pre-trained bars for legend
for i, task in enumerate(tasks):
    # Pre-trained bars
    pretrained_bars[task] = ax.bar(x[i], results[task][0], width, 
                                  label=f'{task} Head', 
                                  color=pretrained_colors[task])
    # Scratch bars
    scratch_bar = ax.bar(x[i] + width, results[task][1], width,
                        label='Tail' if i == 0 else "", 
                        color='pink')

# Customize chart
ax.set_ylabel('Accuracy', fontsize=20)

# Set x-ticks
ax.set_xticks(x + width/2)
ax.set_xticklabels(tasks, fontsize=20)

# Set y-axis limits
ax.set_ylim(0.4, 0.9)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=20)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

for rect in ax.patches:
    autolabel([rect])

# Create legend with all pre-trained models and scratch
legend_elements = [pretrained_bars[task] for task in tasks]
legend_elements.append(scratch_bar)
legend_labels = [f'{task} Head' for task in tasks] + ['Tail']
ax.legend(legend_elements, legend_labels, fontsize=14, loc='upper right', bbox_to_anchor=(1.00, 1.00))

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('Frequency_only_grouped_tasks.png', dpi=300, bbox_inches='tight')