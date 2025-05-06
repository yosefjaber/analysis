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

fig, (ax_box, ax_tbl) = plt.subplots(2, 1, figsize=(8, 6),      
                                     gridspec_kw={'height_ratios':[3, 1]})  

ax_box.boxplot([small, medium, large],                     
               vert=False,
               labels=["Small", "Medium", "Large"],
               boxprops=dict(linewidth=3),
               whiskerprops=dict(linewidth=3),
               capprops=dict(linewidth=3),
               medianprops=dict(linewidth=3),
               flierprops=dict(marker='o', markersize=12,
                               markerfacecolor='none',
                               markeredgecolor='black',
                               markeredgewidth=2))

ax_box.set_xlabel("MSE", labelpad=50)                        
ax_box.set_ylabel("Size")                                    
ax_box.tick_params(axis='both', which='both', width=2, length=8)  

ax_tbl.axis('off')                                       
ax_tbl.table(cellText=summary.values,
             rowLabels=summary.index,
             colLabels=summary.columns,
             cellLoc='center', rowLoc='center',
             loc='center')                                    

fig.tight_layout(pad=2)                                    
tbl = ax_tbl.table(cellText=summary.values,
                   rowLabels=summary.index,
                   colLabels=summary.columns,
                   cellLoc='center', rowLoc='center',
                   loc='center')

tbl.auto_set_font_size(False)  
tbl.set_fontsize(22)          
plt.show()