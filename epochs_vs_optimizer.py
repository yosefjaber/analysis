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

#Epochs vs Optimizer
_100_Adam = 0
_100_AdamW = 0

_250_Adam = 0
_250_AdamW = 0

_500_Adam = 0
_500_AdamW = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name) and re.search("AdamW", model_name):
        _100_Adam += mse_value
        count += 1
    elif re.search("100", model_name) and re.search("Adam", model_name):
        _100_AdamW += mse_value

    elif re.search("250", model_name) and re.search("AdamW", model_name):
        _250_Adam += mse_value
    elif re.search("250", model_name) and re.search("Adam", model_name):
        _250_AdamW += mse_value

    elif re.search("500", model_name) and re.search("AdamW", model_name):
        _500_Adam += mse_value
    elif re.search("500", model_name) and re.search("Adam", model_name):
        _500_AdamW += mse_value
        
    else:
        print("ERROR")
        
_100_Adam /= count
_100_AdamW /= count

_250_Adam /= count
_250_AdamW /= count

_500_Adam /= count
_500_AdamW /= count

values = [_100_Adam, _100_AdamW, _250_Adam, _250_AdamW, _500_Adam, _500_AdamW]
categories = ["100 Adam", "100 AdamW", "250 Adam", "250 AdamW", "500 Adam", "500 AdamW"]

# Group size and spacing
group_size = 2  # Space after every 2 bars
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
plt.title("Epochs relation to Optimizer")
# plt.xlabel("Models")
plt.ylabel("MSE")
plt.show()