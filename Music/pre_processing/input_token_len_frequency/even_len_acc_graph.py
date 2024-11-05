import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Function to read token range and count information from a text file
def read_token_range_data(text_file):
    token_range_data = {}
    with open(text_file, 'r') as file:
        for line in file:
            range_part, count_part = line.split(':')
            token_range = int(range_part.strip())
            count = int(count_part.strip())
            token_range_data[token_range] = count
    return token_range_data

# Function to read model performance data from a JSON file
def read_model_performance_data(json_file):
    with open(json_file, 'r') as file:
        model_performance_data = json.load(file)
    return model_performance_data

# Function to generate and save the plot for each model
def generate_and_save_plot(model_name, token_range_data, model_performance_data, output_dir, model_replace):
    # Preparing the data for plotting
    x_points = [64, 128, 256, 384, 512]
    counts = [sum(token_range_data.get(i, 0) for i in range(x-63, x+1)) for x in x_points]
    
    # Performance scores and lengths
    performance_lengths = list(map(int, model_performance_data.keys()))
    performance_scores = list(model_performance_data.values())

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for token distribution
    ax1.bar(x_points, counts, width=50, color='lightblue', alpha=0.6)
    ax1.set_xlabel('Token Lengths')
    ax1.set_ylabel('Count', color='lightblue')
    ax1.tick_params(axis='y', labelcolor='lightblue')

    # Set x-axis ticks and labels
    ax1.set_xticks(x_points)
    ax1.set_xticklabels(x_points)

    # Creating a second y-axis for model performance
    ax2 = ax1.twinx()
    ax2.plot(performance_lengths, performance_scores, color='orange', marker='o', label='Model Performance')
    ax2.set_ylabel('Model Performance', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add title and grid for better readability
    plt.title(f'Comparison of Model Performance and Token Length Distribution - {model_name}')
    plt.grid(True)

    # Adjust x-axis limits to show all bars
    ax1.set_xlim(0, 576)  # Extend a bit beyond 512 to show the last bar fully

    # Save the plot as PNG in the specified output directory
    output_path = os.path.join(output_dir, f"{model_replace}_performance.png")
    plt.savefig(output_path, bbox_inches='tight')
    
    # Close the plot to avoid memory issues during multiple saves
    plt.close()

# List of models to process
models = [
    "bert-base-uncased",
    "bert-base-chinese",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-german-cased",
    "neuralmind/bert-base-portuguese-cased",
    "tohoku-nlp/bert-base-japanese",
    "microsoft/codebert-base-mlm",
    "neulab/codebert-javascript",
    "neulab/codebert-java",
    "neulab/codebert-python",
    "neulab/codebert-c"
]

# Loop through each model and generate plots
for model in models:
    model_replace = model.replace("/", "_")
    # Define the directory where each model's result files are stored (e.g., {model_name}_result)
    result_dir = f"{model_replace}_result"
    
    # Define paths to the token range text file and performance JSON file
    token_range_file = os.path.join(result_dir, 'token_counts_ranges.txt')
    performance_json_file = os.path.join(result_dir, 'performance_results.json')

    # Read the data from the respective files
    token_range_data = read_token_range_data(token_range_file)
    model_performance_data = read_model_performance_data(performance_json_file)

    # Output directory where results will be saved (you can change this if needed)
    output_dir = result_dir

    # Generate and save the plot for this model
    generate_and_save_plot(model, token_range_data, model_performance_data, output_dir, model_replace)

print("Plots generated and saved successfully.")