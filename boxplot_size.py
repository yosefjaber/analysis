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

small = []
medium = []
large = []

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("small", model_name):
        small.append(mse_value)
    elif re.search("medium", model_name):
        medium.append(mse_value)
    elif re.search("large", model_name):
        large.append(mse_value)
      

category = ["Small", "Medium", "Large"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [small, medium, large],   # a list‑of‑lists
    vert=False,                          # horizontal boxes
    labels=["Small", "Medium", "Large"]
)

plt.title("Relationship of Size and MSE")
plt.xlabel("MSE")
plt.tight_layout()
# plt.show()

amongus = pd.DataFrame(small)
print(amongus.describe())




