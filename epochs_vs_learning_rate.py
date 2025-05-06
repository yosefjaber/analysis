import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MultipleLocator 

results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]

plt.rcParams.update({
    "font.size": 24,          # base size for everything
    "axes.titlesize": 28,     # title font
    "axes.labelsize": 26,     # x‑/y‑label font
    "xtick.labelsize": 24,    # tick labels
    "ytick.labelsize": 24,
    "axes.linewidth": 1
})



#Epochs vs Learning Rate
_100_0_001 = 0
_100_0_0005 = 0
_100_0_0001 = 0
_100_0_00005 = 0

_250_0_001 = 0
_250_0_0005 = 0
_250_0_0001 = 0
_250_0_00005 = 0

_500_0_001 = 0
_500_0_0005 = 0
_500_0_0001 = 0
_500_0_00005 = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name) and re.search("0.001", model_name):
        _100_0_001 += mse_value
        count += 1
    elif re.search("100", model_name) and re.search("0.0005", model_name):
        _100_0_0005 += mse_value
    elif re.search("100", model_name) and re.search("0.0001", model_name):
        _100_0_0001 += mse_value
    elif re.search("100", model_name) and re.search("5e-05", model_name):
        _100_0_00005 += mse_value

    elif re.search("250", model_name) and re.search("0.001", model_name):
        _250_0_001 += mse_value
    elif re.search("250", model_name) and re.search("0.0005", model_name):
        _250_0_0005 += mse_value
    elif re.search("250", model_name) and re.search("0.0001", model_name):
        _250_0_0001 += mse_value
    elif re.search("250", model_name) and re.search("5e-05", model_name):
        _250_0_00005 += mse_value

    elif re.search("500", model_name) and re.search("0.001", model_name):
        _500_0_001 += mse_value
    elif re.search("500", model_name) and re.search("0.0005", model_name):
        _500_0_0005 += mse_value
    elif re.search("500", model_name) and re.search("0.0001", model_name):
        _500_0_0001 += mse_value
    elif re.search("500", model_name) and re.search("5e-05", model_name):
        _500_0_00005 += mse_value
    else:
        print("ERROR")
        
_100_0_001 /= count
_100_0_0005 /= count
_100_0_0001 /= count
_100_0_00005 /= count

_250_0_001 /= count
_250_0_0005 /= count
_250_0_0001 /= count
_250_0_00005 /= count

_500_0_001 /= count
_500_0_0005 /= count
_500_0_0001 /= count
_500_0_00005 /= count


# x = [_500_0_001, _500_0_0005, _500_0_0001, _500_0_00005, _250_0_001, _250_0_0005, _250_0_0001, _250_0_00005, _100_0_001, _100_0_0005, _100_0_0001, _100_0_00005]
# for i in range(len(x)):
#     print(x[i])



top = np.array([_500_0_001, _500_0_0005, _500_0_0001, _500_0_00005])
middle = np.array([_250_0_001, _250_0_0005, _250_0_0001, _250_0_00005])
bottom = np.array([_100_0_001, _100_0_0005, _100_0_0001, _100_0_00005])

grid = np.array([top, middle, bottom])

fig, ax = plt.subplots()

img  = ax.imshow(grid, cmap='viridis', aspect='auto')          # draw ONCE
cbar = fig.colorbar(img, ax=ax)                                # draw ONCE
cbar.locator = MultipleLocator(0.05)                           # ≤— tick step
cbar.update_ticks()

ax.set(
    title   = "Learning Rate vs Epochs Heatmap against MSE",
    xlabel  = "Learning Rate",
    ylabel  = "Epochs",
    xticks  = np.arange(4),
    xticklabels = ["0.001", "0.0005", "0.0001", "0.00005"],
    yticks  = np.arange(3),
    yticklabels = ["500", "250", "100"],
)

plt.tight_layout()
plt.show()
