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

print()

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

print("\n100 Epochs Distribution: ")
x = pd.DataFrame(epoch_100)
print(x.describe())

print("\n250 Epoch Distribution: ")
x = pd.DataFrame(epoch_250)
print(x.describe())

print("\n500 Epoch Distribution: ")
x = pd.DataFrame(epoch_500)
print(x.describe())

print()

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

print("\n0.001 Learning Rate Distribution: ")
x = pd.DataFrame(lr_1)
print(x.describe())

print("\n0.0005 Learning Rate Distribution: ")
x = pd.DataFrame(lr_2)
print(x.describe())

print("\n0.0001 Learning Rate Distribution: ")
x = pd.DataFrame(lr_3)
print(x.describe())

print("\n5e-05 Learning Rate Distribution: ")
x = pd.DataFrame(lr_4)
print(x.describe())

print()

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

print("\nAdam Optimizer Distribution: ")
x = pd.DataFrame(adam)
print(x.describe())

print("\nAdamW Optimizer Distribution: ")
x = pd.DataFrame(adamW)
print(x.describe())

print()

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

print("\nLeft Pyramid Distribution: ")
x = pd.DataFrame(left)
print(x.describe())

print("\nRight Pyramid Distribution: ")
x = pd.DataFrame(right)
print(x.describe())

print("\nDiamond Distribution: ")
x = pd.DataFrame(diamond)
print(x.describe())

print("\nBlock Distribution: ")
x = pd.DataFrame(block)
print(x.describe())

print()

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

print("\nSmall Distribution: ")
x = pd.DataFrame(small)
print(x.describe())

print("\nMedium Distribution: ")
x = pd.DataFrame(medium)
print(x.describe())

print("\nLarge Distribution: ")
x = pd.DataFrame(large)
print(x.describe())
