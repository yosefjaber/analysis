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
      

category = ["100 Epochs", "250 Epochs", "500 Epochs"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [epoch_100, epoch_250, epoch_500],   # a list‑of‑lists
    vert=False,                          # horizontal boxes
    labels=["100 Epochs", "250 Epochs", "500 Epochs"]
)

plt.title("Relationship of Epochs and MSE")
plt.xlabel("MSE")
plt.tight_layout()
# plt.show()

amongus = pd.DataFrame(epoch_100)
print(amongus.describe())

amongus2 = pd.DataFrame(epoch_250)
print(amongus2.describe())

amongus3 = pd.DataFrame(epoch_500)
print(amongus3.describe())