import pandas as pd
import matplotlib.pyplot as plt
import re

# Function to read the data from a tab-separated log file
def extract_data_tsv_v2(file_path):
    data = pd.read_csv(file_path, sep='\t')
    epoch = data['Epoch']
    train_mse = data['train mse']
    # Handling different column names for validation loss
    if '0-29_r2' in data.columns:
        valid_loss = data['0-29_r2']
    elif '0-29_mse' in data.columns:
        valid_loss = data['0-29_mse']
    else:
        valid_loss = None
    return epoch, train_mse, valid_loss

# File paths
file_paths = [
    "multistep0.log",
    "multistep1.log",
    "multistep2.log",
    "multistep3.log",
    "multistep4.log"
]

# Colors for each file
colors = ['b', 'g', 'r', 'c', 'm']  # Blue, Green, Red, Cyan, Magenta

# Plotting
plt.figure(figsize=(12, 6))
for i, file_path in enumerate(file_paths):
    epoch, train_mse, valid_loss = extract_data_tsv_v2(file_path)
    if valid_loss is not None:
        # Dropping the first 5 epochs
        epoch = epoch[5:]
        train_mse = train_mse[5:]
        valid_loss = valid_loss[5:]

        train_color = colors[len(file_paths) - 1 - i]
        valid_color = colors[len(file_paths) - 1 - i] + '--'  # Dashed line for validation loss

        m = re.search("(\d)\.log", file_path)
        plt.plot(epoch, train_mse, train_color, label=f'Multistep {m.group(1)} Train MSE')
        plt.plot(epoch, valid_loss, valid_color, label=f'Multistep {m.group(1)} Valid Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Training and Validation Loss (Multistep 0-4, Epochs 6+)')
plt.yscale('log')  # Logarithmic scale
plt.legend()
plt.grid(True)
plt.show()

