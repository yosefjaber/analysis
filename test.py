import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Increase global font sizes
plt.rcParams.update({'font.size': 26})

# Load your CSV file
df = pd.read_csv("results.csv")

# --- Parse values from Model string ---
df['shape'] = df['Model'].apply(lambda x: re.findall(r'_(left|right|diamond|block)_', x)[0])
df['epochs'] = df['Model'].apply(lambda x: int(re.findall(r'_(\d+)_', x)[0]))

# Function to extract learning rate (handles scientific notation like 5e-05)
def extract_lr(model):
    match = re.search(r'_(\d\.\d+|[1-9]e-\d+)_', model)
    return float(match.group(1)) if match else None

df['lr'] = df['Model'].apply(extract_lr)
df.rename(columns={"MSE": "mse"}, inplace=True)

# Define desired shape order (desired top-to-bottom order: left, right, diamond, block)
desired_order = ["left", "right", "diamond", "block"]

# Set custom shape order for the DataFrame column
df['shape'] = pd.Categorical(df['shape'], categories=desired_order, ordered=True)

# --- Plot: Create a 2x2 grid of heatmaps (one per shape) ---
shapes = desired_order  # or: df['shape'].cat.categories.tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axes = axes.flatten()

for i, shape in enumerate(shapes):
    # Filter by shape
    sub_df = df[df["shape"] == shape]
    # Pivot with learning rate as rows and epochs as columns
    pivot = sub_df.pivot_table(index="lr", columns="epochs", values="mse")
    
    sns.heatmap(
        pivot,
        ax=axes[i],
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar=True,
        annot_kws={'size': 20}  # Increase annotation font size
    )
    axes[i].set_title(f"Shape = {shape}", fontsize=24)
    axes[i].set_xlabel("Epochs", fontsize=24)
    axes[i].set_ylabel("Learning Rate", fontsize=24)
    axes[i].tick_params(axis='both', which='major', labelsize=22)

# Remove any unused subplots (if there are less than 4 shapes)
for j in range(i + 1, 4):
    fig.delaxes(axes[j])

plt.suptitle("MSE Heatmaps by Learning Rate & Epochs (Grouped by Shape)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
