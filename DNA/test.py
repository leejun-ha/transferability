import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['H4', 'H3K9ac', 'Protein', 'Music']
models = ['BERT', 'CodeBERT']

# Results: [pre-trained BERT, pre-trained CodeBERT], [scratch BERT, scratch CodeBERT]
results = {
    'H4':     [[0.7242, 0.7040], [0.6178, 0.6901]],
    'H3K9ac': [[0.8360, 0.8640], [0.6247, 0.8417]],
    'Protein':[[0.6380, 0.6633], [0.5760, 0.6392]],
    'Music':  [[0.4825, 0.4801], [0.4161, 0.4470]]
}

# Set up bar positions
width = 0.25  # Bar width
task_spacing = 0.4  # Space between task groups
model_spacing = 1.2  # Space between BERT and CodeBERT sections

# Calculate base positions for BERT and CodeBERT sections
bert_base = 0
codebert_base = bert_base + 4 * (2 * width + task_spacing) + model_spacing

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(15, 6))

# Define colors for pre-trained tasks
pretrained_colors = {
    'H4': '#ff9999',      # Mild red
    'H3K9ac': '#ff9999',  # Mild red
    'Protein': 'green',
    'Music': '#1f77b4'    # Blue
}

# Plot bars for BERT and CodeBERT
scratch_bar = None  # To store scratch bar for legend
pretrained_bars = {}  # To store pre-trained bars for legend

for i, task in enumerate(tasks):
    # BERT section
    bert_pos = bert_base + i * (2 * width + task_spacing)
    pretrained_bars[task] = ax.bar(bert_pos, results[task][0][0], width, 
                                  label=f'{task} Pre-Trained', 
                                  color=pretrained_colors[task])
    scratch_bar = ax.bar(bert_pos + width, results[task][1][0], width,
                        label='Scratch' if i == 0 else "", 
                        color='orange')
    
    # CodeBERT section
    codebert_pos = codebert_base + i * (2 * width + task_spacing)
    ax.bar(codebert_pos, results[task][0][1], width,
           color=pretrained_colors[task])
    ax.bar(codebert_pos + width, results[task][1][1], width,
           color='orange')

# Customize chart
ax.set_ylabel('Accuracy', fontsize=18)
ax.set_title('Comparison of Task Results for BERT and CodeBERT', fontsize=18)

# Calculate the center position for each group of bars
bert_center = bert_base + (4 * (2 * width + task_spacing) - task_spacing) / 2
codebert_center = codebert_base + (4 * (2 * width + task_spacing) - task_spacing) / 2

# Set x-ticks to center model labels under their respective groups
ax.set_xticks([bert_center, codebert_center])
ax.set_xticklabels(models, fontsize=14)

# Set y-axis limits
ax.set_ylim(0.4, 0.9)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for rect in ax.patches:
    autolabel([rect])

# Create legend with all pre-trained models and scratch
legend_elements = [pretrained_bars[task] for task in tasks]
legend_elements.append(scratch_bar)
legend_labels = [f'{task} Pre-Trained' for task in tasks] + ['Scratch']
ax.legend(legend_elements, legend_labels, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('bert_codebert_grouped_tasks_centered_labels.png', dpi=300, bbox_inches='tight')