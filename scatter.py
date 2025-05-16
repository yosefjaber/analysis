import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1,
    "figure.dpi": 150
})

results = pd.read_csv("results.csv")
MAE = results["MAE"]
CV  = results["CV"]

fig, ax = plt.subplots(figsize=(6,4))

# keep the log–log scaling
ax.set_xscale('log')
ax.set_yscale('log')

ax.scatter(CV, MAE, s=3)
ax.set_xlabel("Mean Standard Error")
ax.set_ylabel("Mean Absolute Error")

# ------------------------------------------------------------------
# 1) put ticks where you want them
#
#    Here I put one every 0.1 from 0.9–2.0 –– adjust to taste.
# ------------------------------------------------------------------
xticks = np.arange(0.9, 2.05, 0.1)
yticks = np.arange(0.3, 1.05, 0.1)

ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))

# ------------------------------------------------------------------
# 2) tell Matplotlib **exactly** how to print each tick
# ------------------------------------------------------------------
fmt = mticker.FuncFormatter(lambda x, pos: f"{x:.2f}")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

plt.tight_layout()
plt.show()
