import pandas as pd
import re
import matplotlib.pyplot as plt

# Global font sizes
plt.rcParams.update({
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})

# Load & parse
df = pd.read_csv('results.csv')
df.columns = df.columns.str.lower().str.strip()

sizes, shapes, epochs, lrs, optimizers = [], [], [], [], []
for model_name in df["model"]:
    sizes.append("small"  if "small"  in model_name else
                 "medium" if "medium" in model_name else
                 "large"  if "large"  in model_name else "unknown")
    match_shape  = re.search(r'layers_(.*?)_\d+', model_name)
    match_epochs = re.search(r'_(\d+)_',      model_name)
    match_lr     = re.search(r'_(0\.\d+)_',   model_name)
    shapes.append(match_shape.group(1).replace('_',' ') if match_shape else "unknown")
    epochs.append(int(match_epochs.group(1))      if match_epochs else -1)
    lrs.append(float(match_lr.group(1))           if match_lr else 5e-05)
    optimizers.append("AdamW" if "AdamW" in model_name else "Adam")

df["size"]         = sizes
df["shapetype"]    = shapes
df["epochs"]       = epochs
df["learningrate"] = lrs
df["optimizer"]    = optimizers

df['xgroup'] = df['size'] + ' | ' + df['shapetype']
df['ygroup'] = df['epochs'].astype(str) + ' | ' + df['learningrate'].astype(str)

# 2) Compute global color range
global_min = df['mse'].min()
global_max = df['mse'].max()

# 3) Plot with the same vmin/vmax
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, constrained_layout=True)
for ax, opt in zip(axes, ['Adam', 'AdamW']):
    ax.set_title(opt)
    subset = df[df['optimizer'] == opt]
    pivot = subset.pivot(index='ygroup', columns='xgroup', values='mse')
    im = ax.imshow(
        pivot.values,
        aspect='auto',
        vmin=global_min,
        vmax=global_max
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, rotation=0)
    ax.set_xlabel('Size | ShapeType')
    ax.set_ylabel('Epochs | LearningRate')

    # individual colorbar
    cbar = fig.colorbar(im, ax=ax, label='MSE')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('MSE', fontsize=8)

# plt.tight_layout()
plt.savefig("heatmaps.png", dpi=300)
plt.show()