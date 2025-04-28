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


fig, (ax_box, ax_tbl) = plt.subplots(2, 1, figsize=(8, 6),        
                                     gridspec_kw={'height_ratios':[3, 1]})  

ax_box.boxplot([epoch_100, epoch_250, epoch_500],                            
               vert=False,
               labels=["100", "250", "500"],
               boxprops=dict(linewidth=3),
               whiskerprops=dict(linewidth=3),
               capprops=dict(linewidth=3),
               medianprops=dict(linewidth=3),
               flierprops=dict(marker='o', markersize=12,
                               markerfacecolor='none',
                               markeredgecolor='black',
                               markeredgewidth=2))

ax_box.set_xlabel("MSE", labelpad=0)                             
ax_box.set_ylabel("Epochs")                                         
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