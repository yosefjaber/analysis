import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "font.size": 24,          # base size for everything
    "axes.titlesize": 28,     # title font
    "axes.labelsize": 26,     # x‑/y‑label font
    "xtick.labelsize": 24,    # tick labels
    "ytick.labelsize": 24,
    "axes.linewidth": 3 
})

results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]

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


gap = .35                          # distance you want between boxes
pos = [1, 1+gap, 1+2*gap]          # 1 → 1.35 → 1.70

fig, (ax_box, ax_tbl) = plt.subplots(
    2, 1, figsize=(8, 5),
    gridspec_kw={'height_ratios':[3,1], 'hspace':0.25}
)

# --- compact box‑plot -------------------------------------------------
ax_box.boxplot(
    [epoch_100, epoch_250, epoch_500],
    vert=False,
    positions=pos,                 # <<–––– here
    widths=gap*0.65,               # a bit narrower than the gap
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
ax_box.set_ylabel('Epochs')

ax_box.set_yticks(pos)
ax_box.set_yticklabels(['100', '250', '500'])

# keep just a sliver of padding above & below the outer boxes
ax_box.set_ylim(pos[0]-gap*0.6, pos[-1]+gap*0.6)

# ---------------------------------------------------------------------
ax_tbl.axis('off')
tbl = ax_tbl.table(cellText=summary.values,
                   rowLabels=summary.index,
                   colLabels=summary.columns,
                   cellLoc='center', rowLoc='center',
                   loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(22)

plt.show()
