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

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Average MSE Heatmap")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.xticks(np.arange(4), ["0.001", "0.0005", "0.0001", "0.00005"])
plt.yticks(np.arange(3), ["500", "250", "100"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)
plt.show()