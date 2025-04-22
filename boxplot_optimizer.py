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
})


results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]

adam = []
adamW = []

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("AdamW", model_name):
        adamW.append(mse_value)
    elif re.search("Adam", model_name):
        adam.append(mse_value)
      

category = ["Adam", "AdamW"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [adam, adamW],   # a list‑of‑lists
    vert=False,                          # horizontal boxes
    labels=["Adam", "AdamW"],
    boxprops=dict(linewidth=3),
        whiskerprops=dict(linewidth=3),
        capprops=dict(linewidth=3),
        medianprops=dict(linewidth=3)
)

plt.title("Relationship of Optimizer and MSE")
plt.xlabel("MSE")
plt.tight_layout()
plt.tick_params(axis='both', which='both', width=2, length=8)
plt.show()

amongus = pd.DataFrame(adam)
print(amongus.describe())

amongus2 = pd.DataFrame(adamW)
print(amongus2.describe())




