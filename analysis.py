import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

results = pd.read_csv("results.csv")
Models = results["Model"]
MSE = results["MSE"]
R2 =  results["R^2"]
MAE =  results["MAE"]
CV =  results["CV"]
Count =  results["Count"]

# print(MSE.describe())


mse_dict = {}
for i in range(len(MSE)):
    mse_dict[Models[i]] = MSE[i]

sorted_mse_dict = dict(sorted(mse_dict.items(), key=lambda item: item[1]))

first_ten = dict(list(sorted_mse_dict.items())[10:])
# for key in first_ten:
#     print(key)

plt.boxplot(MSE, vert=False)
plt.title("MSE boxplot")
plt.xlabel("MSE")
plt.yticks([1], [""])  # Remove the '1' label
plt.show()