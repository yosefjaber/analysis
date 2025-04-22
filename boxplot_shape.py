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

left = []
right = []
diamond = []
block = []

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("left", model_name):
        left.append(mse_value)
    elif re.search("right", model_name):
        right.append(mse_value)
    elif re.search("diamond", model_name):
        diamond.append(mse_value)
    elif re.search("block", model_name):
        block.append(mse_value)
      

category = ["Left Pyramid", "Right Pyramid", "Diamond", "Block"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [left, right, diamond, block],   # a list‑of‑lists
    vert=False,                          # horizontal boxes
    labels=["Left Pyramid", "Right Pyramid", "Diamond", "Block"]
)

plt.title("Relationship of Shape and MSE")
plt.xlabel("MSE")
plt.tight_layout()
# plt.show()

def IQR(x):
    Q1 = x.quantile(0.25)
    Q2 = x.quantile(0.75)
    print(Q2-Q1)

amongus = pd.DataFrame(left)
# print(amongus.describe())
IQR(amongus)

amongus2 = pd.DataFrame(right)
# print(amongus2.describe())
IQR(amongus2)

amongus3 = pd.DataFrame(diamond)
# print(amongus3.describe())
IQR(amongus3)

amongus4 = pd.DataFrame(block)
# print(amongus3.describe())
IQR(amongus4)



