import matplotlib.pyplot as plt
import numpy as np
# H4    [0.7242, 0.7040] [0.6178, 0.6901]
# H3K9ac    [0.8360, 0.8640] [0.6247, 0.8417]
# Protein   [0.6380, 0.6633] [0.5760, 0.6392]
# Music [0.4825, 0.4801 ] [0.4161, 0.4470]
# all_results_avg [0.67, 0.67] [0.56, 0.65]
# freq_avg [0.7221, 0.7247] [0.6956, 0.6947]
# Data
models = ['BERT', 'CodeBERT']
pre_training_results = [0.7221, 0.7247]
scratch_results = [0.6956, 0.6947]

# Set up the bar positions
x = np.arange(len(models))
width = 0.2  # Width of each bar

# Adjust the distance between groups
group_distance = 0.2  # Decrease this value to bring groups closer

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# Create the bars with adjusted positions
rects1 = ax.bar(x - width/2, pre_training_results, width, label='Head', color='blue') # basic blue hex code
rects2 = ax.bar(x + width/2, scratch_results, width, label='Tail', color='pink')

# Customize the chart
ax.set_ylabel('Accuracy', fontsize=18)
# ax.set_title('Comparison of Pre-Training and Scratch Results', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=30)
ax.legend(fontsize=16, loc='upper right')

# Set y-axis limits
ax.set_ylim(0.5, 0.8)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=20)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)

autolabel(rects1)
autolabel(rects2)

# Adjust the x-axis to bring groups closer
ax.set_xlim(-0.5, len(models) - 0.5)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('Head_tail_compare.png', dpi=300)

