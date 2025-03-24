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

values = [_100_left, _100_right, _100_diamond, _100_block, _250_left, _250_right, _250_diamond, _250_block, _500_left, _500_right, _500_diamond, _500_block]
categories = ["100 left", "100 right", "100 diamond", "100 block", "250 left", "250 right", "250 diamond", "250 block", "500 left", "500 right", "500 diamond", "500 block"]

# Group size and spacing
group_size = 4  # Space after every 4 bars
spacer = 1      # Space width

# Adjust x positions to include spacing
x_positions = []
for i in range(len(values)):
    x_positions.append(i + (i // group_size) * spacer)

# Plotting the bar graph
plt.figure(figsize=(12, 6))
plt.bar(x_positions, values, color='skyblue', width=0.8)

# Adjust x-ticks to align with categories
plt.xticks(x_positions, categories, rotation=45, ha="right")
plt.title("Epochs relation to Shape")
plt.ylabel("MSE")
plt.show()