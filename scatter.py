import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9, #Nothing
    "axes.labelsize": 8.6, #Bottom MSE text
    "xtick.labelsize": 8, #x tick marks
    "ytick.labelsize": 8, #y tick marks
    "axes.linewidth": 1, #Rectangle Border
    "figure.dpi": 300
})

results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]

plt.scatter(MSE, MAE, s=3)
plt.xlabel("Mean Standard Error")
plt.ylabel("Mean Absolute Error")

plt.show()

correlation = MSE.corr(MAE)
print("Correlation coefficient:", correlation)
