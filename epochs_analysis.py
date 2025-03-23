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


# x = [_500_0_001, _500_0_0005, _500_0_0001, _500_0_00005, _250_0_001, _250_0_0005, _250_0_0001, _250_0_00005, _100_0_001, _100_0_0005, _100_0_0001, _100_0_00005]
# for i in range(len(x)):
#     print(x[i])



top = np.array([_500_small, _500_medium, _500_large])
middle = np.array([_250_small, _250_medium, _250_large])
bottom = np.array([_100_small, _100_medium, _100_large])

grid = np.array([top, middle, bottom])

print(count)

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Average MSE Heatmap")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.xticks(np.arange(3), ["small", "medium", "large"])
plt.yticks(np.arange(3), ["500", "250", "100"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)
plt.show()





#Epochs vs Shape
_100_left = 0
_100_right = 0
_100_diamond = 0
_100_block = 0

_250_left = 0
_250_right = 0
_250_diamond = 0
_250_block = 0

_500_left = 0
_500_right = 0
_500_diamond = 0
_500_block = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name) and re.search("left", model_name):
        _100_left += mse_value
        count += 1
    elif re.search("100", model_name) and re.search("right", model_name):
        _100_right += mse_value
    elif re.search("100", model_name) and re.search("diamond", model_name):
        _100_diamond += mse_value
    elif re.search("100", model_name) and re.search("block", model_name):
        _100_block += mse_value

    elif re.search("250", model_name) and re.search("left", model_name):
        _250_left += mse_value
    elif re.search("250", model_name) and re.search("right", model_name):
        _250_right += mse_value
    elif re.search("250", model_name) and re.search("diamond", model_name):
        _250_diamond += mse_value
    elif re.search("250", model_name) and re.search("block", model_name):
        _250_block += mse_value

    elif re.search("500", model_name) and re.search("left", model_name):
        _500_left += mse_value
    elif re.search("500", model_name) and re.search("right", model_name):
        _500_right += mse_value
    elif re.search("500", model_name) and re.search("diamond", model_name):
        _500_diamond += mse_value
    elif re.search("500", model_name) and re.search("block", model_name):
        _500_block += mse_value
    else:
        print("ERROR")
        
_100_left /= count
_100_right /= count
_100_diamond /= count
_100_block /= count

_250_left /= count
_250_right /= count
_250_diamond /= count
_250_block /= count

_500_left /= count
_500_right /= count
_500_diamond /= count
_500_block /= count


# x = [_500_0_001, _500_0_0005, _500_0_0001, _500_0_00005, _250_0_001, _250_0_0005, _250_0_0001, _250_0_00005, _100_0_001, _100_0_0005, _100_0_0001, _100_0_00005]
# for i in range(len(x)):
#     print(x[i])



top = np.array([_500_left, _500_right, _500_diamond, _500_block])
middle = np.array([_250_left, _250_right, _250_diamond, _250_block])
bottom = np.array([_100_left, _100_right, _100_diamond, _100_block])

grid = np.array([top, middle, bottom])

print(count)

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Average MSE Heatmap")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.xticks(np.arange(4), ["left", "right", "diamond", "block"])
plt.yticks(np.arange(3), ["500", "250", "100"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)
plt.show()


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

print(count)

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Average MSE Heatmap")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.xticks(np.arange(4), ["0.001", "0.0005", "0.0001", "0.00005"])
plt.yticks(np.arange(3), ["500", "250", "100"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)
# plt.show()