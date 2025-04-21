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


#Learning Rate vs Size
small_0_001 = 0
small_0_0005 = 0
small_0_0001 = 0
small_0_00005 = 0

medium_0_001 = 0
medium_0_0005 = 0
medium_0_0001 = 0
medium_0_00005 = 0

large_0_001 = 0
large_0_0005 = 0
large_0_0001 = 0
large_0_00005 = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("small", model_name) and re.search("0.001", model_name):
        small_0_001 += mse_value
        count += 1
    elif re.search("small", model_name) and re.search("0.0005", model_name):
        small_0_0005 += mse_value
    elif re.search("small", model_name) and re.search("0.0001", model_name):
        small_0_0001 += mse_value
    elif re.search("small", model_name) and re.search("5e-05", model_name):
        small_0_00005 += mse_value

    elif re.search("medium", model_name) and re.search("0.001", model_name):
        medium_0_001 += mse_value
    elif re.search("medium", model_name) and re.search("0.0005", model_name):
        medium_0_0005 += mse_value
    elif re.search("medium", model_name) and re.search("0.0001", model_name):
        medium_0_0001 += mse_value
    elif re.search("medium", model_name) and re.search("5e-05", model_name):
        medium_0_00005 += mse_value

    elif re.search("large", model_name) and re.search("0.001", model_name):
        large_0_001 += mse_value
    elif re.search("large", model_name) and re.search("0.0005", model_name):
        large_0_0005 += mse_value
    elif re.search("large", model_name) and re.search("0.0001", model_name):
        large_0_0001 += mse_value
    elif re.search("large", model_name) and re.search("5e-05", model_name):
        large_0_00005 += mse_value
    else:
        print("ERROR")
        
small_0_001 /= count
small_0_0005 /= count
small_0_0001 /= count
small_0_00005 /= count

medium_0_001 /= count
medium_0_0005 /= count
medium_0_0001 /= count
medium_0_00005 /= count

large_0_001 /= count
large_0_0005 /= count
large_0_0001 /= count
large_0_00005 /= count


# x = [_500_0_001, _500_0_0005, _500_0_0001, _500_0_00005, _250_0_001, _250_0_0005, _250_0_0001, _250_0_00005, _100_0_001, _100_0_0005, _100_0_0001, _100_0_00005]
# for i in range(len(x)):
#     print(x[i])



top = np.array([large_0_001, large_0_0005, large_0_0001, large_0_00005])
middle = np.array([medium_0_001, medium_0_0005, medium_0_0001, medium_0_00005])
bottom = np.array([small_0_001, small_0_0005, small_0_0001, small_0_00005])

grid = np.array([top, middle, bottom])
plt.title("Learning Rate vs Size Heatmap against MSE")
plt.xlabel("Learning Rate")
plt.ylabel("Size")
plt.xticks(np.arange(4), ["0.001", "0.0005", "0.0001", "0.00005"])
plt.yticks(np.arange(3), ["Large", "Medium", "Small"])
# plt.grid(visible=True, color='white', linestyle='-', linewidth=0.5)

im = plt.imshow(grid, cmap='hot', interpolation='nearest')
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=26)  # Increase colorbar tick label size


plt.title("Learning Rate vs Size Heatmap against MSE", fontsize=30)
plt.xlabel("Learning Rate", fontsize=26)
plt.ylabel("Size", fontsize=26)
plt.xticks(np.arange(4), ["0.001", "0.0005", "0.0001", "0.00005"], fontsize=26)
plt.yticks(np.arange(3), ["Large", "Medium", "Small"], fontsize=26)

plt.show()