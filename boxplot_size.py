import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.linewidth": 3
})

results = pd.read_csv("results.csv")
small, medium, large = [], [], []

for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value  = results.iloc[i]["MSE"]
    if   re.search("small",  model_name): small.append(mse_value)
    elif re.search("medium", model_name): medium.append(mse_value)
    elif re.search("large",  model_name): large.append(mse_value)

summary = (pd.DataFrame({"Small":  pd.Series(small).describe(),
                         "Medium": pd.Series(medium).describe(),
                         "Large":  pd.Series(large).describe()})
           .loc[['count','mean','std','min','25%','50%','75%','max']]
           .T.round(3))

gap = .0001                      # distance you want between boxes
pos = [1, 1+gap, 1+2*gap]          # 1 → 1.35 → 1.70

fig, (ax_box, ax_tbl) = plt.subplots(
    2, 1, figsize=(8, 5),
    gridspec_kw={'height_ratios':[0.9,1]}
)

ax_box.boxplot(
    [small, medium, large],
    vert=False,
    positions=pos,                 # <<–––– here
    widths=gap*0.5,               # a bit narrower than the gap
    boxprops     =dict(linewidth=3),
    whiskerprops =dict(linewidth=3),
    capprops     =dict(linewidth=3),
    medianprops  =dict(linewidth=3),
    flierprops   =dict(marker='o', markersize=12,
                       markerfacecolor='none',
                       markeredgecolor='black',
                       markeredgewidth=2)
)
ax_box.set_xlabel('MSE')
ax_box.set_ylabel('Size', labelpad=25)  # Increase from default (usually ~4-10)                    
ax_box.tick_params(axis='both', which='both', width=2, length=8)  

ax_box.set_yticks(pos)
ax_box.set_yticklabels(['Small', 'Medium', 'Large'])
ax_box.set_ylim(pos[0] - gap * 0.6, pos[-1] + gap * 0.6)

# ---------------------------------------------------------------------
ax_tbl.axis('off')
tbl = ax_tbl.table(cellText=summary.values,
                   rowLabels=summary.index,
                   colLabels=summary.columns,
                   cellLoc='center', rowLoc='center',
                   loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(22)

plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()