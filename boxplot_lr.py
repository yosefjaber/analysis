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

lr_1 = []
lr_2 = []
lr_3 = []
lr_4 = []

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("0.001", model_name):
        lr_1.append(mse_value)
    elif re.search("0.0005", model_name):
        lr_2.append(mse_value)
    elif re.search("0.0001", model_name):
        lr_3.append(mse_value)
    elif re.search("5e-05", model_name):
        lr_4.append(mse_value)
      

category = ["0.001", "0.0005", "0.0001", "0.00005"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [lr_1, lr_2, lr_3, lr_4],   # a list‑of‑lists
    vert=True,                          # horizontal boxes
    labels=["0.001", "0.0005", "0.0001", "0.00005"]
)

plt.title("Relationship of Learning Rate and MSE")
plt.xlabel("MSE")
plt.tight_layout()
# plt.show()

amongus = pd.DataFrame(lr_1)
print(amongus.describe())

amongus2 = pd.DataFrame(lr_2)
print(amongus2.describe())

amongus3 = pd.DataFrame(lr_3)
print(amongus3.describe())

amongus3 = pd.DataFrame(lr_4)
print(amongus3.describe())




