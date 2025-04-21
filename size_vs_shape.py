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

epoch_100 = 0
epoch_250 = 0
epoch_500 = 0

count = 0
for i in range(len(results)):
    model_name = results.iloc[i]["Model"]
    mse_value = results.iloc[i]["MSE"]

    if re.search("100", model_name):
        epoch_100 += mse_value
        count += 1
    elif re.search("250", model_name):
        epoch_250 += mse_value
    elif re.search("500", model_name):
       epoch_500 += mse_value

epoch_100 /= count
epoch_250 /= count
epoch_500 /= count

values = [epoch_100, epoch_250, epoch_500]
category = ["100 Epochs", "250 Epochs", "500 Epochs"]

plt.figure(figsize=(12, 6))
plt.bar(category, values, color='skyblue', width=0.8)
plt.title("Relationship of Epochs and MSE")
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()
