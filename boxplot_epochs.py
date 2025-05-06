import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9, #Nothing
    "axes.labelsize": 8.6, #Bottom MSE text
    "xtick.labelsize": 8, #x tick marks
    "ytick.labelsize": 8, #y tick marks
    "axes.linewidth": 1, #Rectangle Border
    "figure.dpi": 300
})

results = pd.read_csv("results.csv")

epoch_100 = []
epoch_250 = []
epoch_500 = []

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name):
        epoch_100.append(mse_value)
    elif re.search("250", model_name):
        epoch_250.append(mse_value)
    elif re.search("500", model_name):
        epoch_500.append(mse_value)
      

summary = (pd.DataFrame({"100":  pd.Series(epoch_100).describe(),
                         "250": pd.Series(epoch_250).describe(),
                         "500":  pd.Series(epoch_500).describe()})
           .loc[['count','mean','std','min','25%','50%','75%','max']]
           .T.round(3))


gap = .0001                      # distance you want between boxes
pos = [1, 1+gap, 1+2*gap]          # 1 → 1.35 → 1.70

fig, (ax_box, ax_tbl) = plt.subplots(
    2, 1, figsize=(8, 5),
    gridspec_kw={'height_ratios':[0.9,1]}
)

# --- compact box‑plot -------------------------------------------------
ax_box.boxplot(
    [epoch_100, epoch_250, epoch_500],
    vert=False,
    positions=pos,                 # <<–––– here
    widths=gap*0.5,               # a bit narrower than the gap
    boxprops     =dict(linewidth=1),
    whiskerprops =dict(linewidth=1),
    capprops     =dict(linewidth=1),
    medianprops  =dict(linewidth=1),
    flierprops   =dict(marker='o', markersize=2,
                       markerfacecolor='none',
                       markeredgecolor='black',
                       markeredgewidth=1)
)

ax_box.set_xlabel('MSE')
ax_box.set_ylabel('Epochs', labelpad=15)  # Increase from default (usually ~4-10)
ax_box.tick_params(axis='both', which='both', width=1, length=4)  

ax_box.set_yticks(pos)
ax_box.set_yticklabels(['100', '250', '500'])
ax_box.set_ylim(pos[0] - gap * 0.6, pos[-1] + gap * 0.6)

# ---------------------------------------------------------------------
ax_tbl.axis('off')
tbl = ax_tbl.table(cellText=summary.values,
                   rowLabels=summary.index,
                   colLabels=summary.columns,
                   cellLoc='center', rowLoc='center',
                   loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()
