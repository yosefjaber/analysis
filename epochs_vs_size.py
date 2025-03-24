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

#Epochs vs Size
_100_small = 0
_100_medium = 0
_100_large = 0

_250_small = 0
_250_medium = 0
_250_large = 0

_500_small = 0
_500_medium = 0
_500_large = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name) and re.search("small", model_name):
        _100_small += mse_value
        print(model_name)
        count += 1
    elif re.search("100", model_name) and re.search("medium", model_name):
        _100_medium += mse_value
    elif re.search("100", model_name) and re.search("large", model_name):
        _100_large += mse_value

    elif re.search("250", model_name) and re.search("small", model_name):
        _250_small += mse_value
    elif re.search("250", model_name) and re.search("medium", model_name):
        _250_medium += mse_value
    elif re.search("250", model_name) and re.search("large", model_name):
        _250_large += mse_value

    elif re.search("500", model_name) and re.search("small", model_name):
        _500_small += mse_value
    elif re.search("500", model_name) and re.search("medium", model_name):
        _500_medium += mse_value
    elif re.search("500", model_name) and re.search("large", model_name):
        _500_large += mse_value
    else:
        print("ERROR")
        
_100_small /= count
_100_medium /= count
_100_large /= count

_250_small /= count
_250_medium /= count
_250_large /= count

_500_small /= count
_500_medium /= count
_500_large /= count

top = np.array([_500_small, _500_medium, _500_large])
middle = np.array([_250_small, _250_medium, _250_large])
bottom = np.array([_100_small, _100_medium, _100_large])

grid = np.array([top, middle, bottom])

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Relationship of Size and Epochs")
plt.xlabel("Size")
plt.ylabel("Epochs")
plt.xticks(np.arange(3), ["Small", "Medium", "Large"])
plt.yticks(np.arange(3), ["500", "250", "100"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)
plt.show()