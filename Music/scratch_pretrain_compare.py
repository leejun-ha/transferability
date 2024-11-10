import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['BERT', 'CodeBERT']
pre_training_results = [0.4901, 0.4941]  # Example values for pre-training
scratch_results = [0.4482, 0.4432]  # Example values for scratch

# Set up the bar positions
x = np.arange(len(models))
width = 0.35

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bars
rects1 = ax.bar(x - width/2, pre_training_results, width, label='Pre-Trained', color='orange')
rects2 = ax.bar(x + width/2, scratch_results, width, label='Scratch', color='blue')

# Customize the chart
ax.set_ylabel('Scores')
ax.set_title('Comparison of Pre-Training and Scratch Results')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_chart.png')

# Display the plot (optional, if you want to see it in the notebook or IDE)
plt.show()
